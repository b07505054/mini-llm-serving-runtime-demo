import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RuntimeSimulateAdapterTest(unittest.TestCase):
    def test_successful_response_passes_through_honest_fields_unchanged(self):
        runtime = server.MiniServingRuntime()
        fake_payload = {
            "request_id": "sim-1",
            "result_type": "simulated",
            "policy": "inflight_paged_kv_continuous_batching",
            "git_commit": "abc1234",
            "ttft_ms": 12.3,
            "tpot_ms": 4.5,
            "e2e_latency_ms": 99.9,
            "kv_page_lifecycle": {
                "usefulness_score": 0.83,
                "usefulness_score_ema": 0.71,
                "adaptive_guard_active": True,
                "adaptive_prefetch_skips": 2,
            },
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(fake_payload)):
            result = runtime.live_runtime_simulate(
                {"prompt_tokens": 256, "max_output_tokens": 32}
            )
        self.assertEqual(result, fake_payload)
        lifecycle = result["kv_page_lifecycle"]
        self.assertEqual(lifecycle["usefulness_score"], 0.83)
        self.assertEqual(lifecycle["usefulness_score_ema"], 0.71)
        self.assertEqual(lifecycle["adaptive_guard_active"], True)

    def test_service_unavailable_returns_status_unavailable_without_raising(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.live_runtime_simulate(
                {"prompt_tokens": 256, "max_output_tokens": 32}
            )
        self.assertEqual(
            result, {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        )

    def test_generate_unaffected_by_runtime_simulate_adapter(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.generate({"prompt_tokens": 128, "max_output_tokens": 16})
        self.assertEqual(result["status"], "completed")
        self.assertIn("ttft_ms", result)
        self.assertIn("tpot_ms", result)

    def test_qwen_ask_unaffected_by_runtime_simulate_adapter(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.ask({"question": "What is the capital of France?"})
        self.assertIn("answer", result)
        self.assertIn("source_status", result)

    def test_adapter_status_and_available_false_when_unreachable(self):
        adapter = server.RuntimeSimulateAdapter(url="http://127.0.0.1:1/simulate", timeout=0.2)
        status = adapter.status()
        self.assertFalse(status["ready"])
        self.assertFalse(adapter.available())

    def test_batch_successful_response_passes_through_unchanged(self):
        runtime = server.MiniServingRuntime()
        fake_payload = {
            "result_type": "simulated",
            "mode": "batch",
            "policy": "inflight_paged_kv_continuous_batching",
            "git_commit": "abc1234",
            "request_count": 3,
            "ttft_ms": 12.3,
            "tpot_ms": 4.5,
            "e2e_latency_ms": 99.9,
            "rejected_requests": 1,
            "oom_events": 1,
            "kv_page_lifecycle": {
                "pressure_prefetch_skips": 7,
                "prefetch_waste": 0,
            },
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(fake_payload)):
            result = runtime.live_runtime_simulate_batch(
                {
                    "requests": [
                        {"prompt_tokens": 256, "max_output_tokens": 32},
                        {"prompt_tokens": 4096, "max_output_tokens": 64},
                        {"prompt_tokens": 4096, "max_output_tokens": 64},
                    ]
                }
            )
        self.assertEqual(result, fake_payload)
        self.assertEqual(result["request_count"], 3)
        self.assertEqual(result["mode"], "batch")

    def test_batch_request_targets_batch_url_not_single_url(self):
        runtime = server.MiniServingRuntime()
        seen_urls = []

        def fake_urlopen(request, timeout=None):
            seen_urls.append(request.full_url)
            return FakeResponse({"result_type": "simulated", "mode": "batch"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            runtime.live_runtime_simulate_batch({"requests": [{"prompt_tokens": 64, "max_output_tokens": 16}]})
        self.assertEqual(seen_urls, [server.RUNTIME_SIMULATE_BATCH_URL])

    def test_batch_service_unavailable_returns_status_unavailable_without_raising(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.live_runtime_simulate_batch(
                {"requests": [{"prompt_tokens": 256, "max_output_tokens": 32}]}
            )
        self.assertEqual(
            result, {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        )

    def test_generate_unaffected_by_runtime_simulate_batch_addition(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.generate({"prompt_tokens": 128, "max_output_tokens": 16})
        self.assertEqual(result["status"], "completed")
        self.assertIn("ttft_ms", result)
        self.assertIn("tpot_ms", result)

    def test_qwen_ask_unaffected_by_runtime_simulate_batch_addition(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.ask({"question": "What is the capital of France?"})
        self.assertIn("answer", result)
        self.assertIn("source_status", result)

    def test_session_create_passes_through_and_targets_session_base_url(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"session_id": "sess-1", "result_type": "simulated"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_runtime_session_create({"total_pages": 32})
        self.assertEqual(seen, [(server.RUNTIME_SESSION_BASE_URL, "POST")])
        self.assertEqual(result["session_id"], "sess-1")

    def test_session_create_unavailable_returns_status_unavailable(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.live_runtime_session_create({})
        self.assertEqual(
            result, {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        )

    def test_session_request_targets_correct_path_and_passes_through(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"admitted": True, "result_type": "simulated"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_runtime_session_request(
                "sess-1", {"prompt_tokens": 256, "max_output_tokens": 32}
            )
        self.assertEqual(
            seen, [(f"{server.RUNTIME_SESSION_BASE_URL}/sess-1/request", "POST")]
        )
        self.assertTrue(result["admitted"])

    def test_session_step_targets_correct_path(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"events": [], "result_type": "simulated"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            runtime.live_runtime_session_step("sess-1", {})
        self.assertEqual(
            seen, [(f"{server.RUNTIME_SESSION_BASE_URL}/sess-1/step", "POST")]
        )

    def test_session_cancel_targets_correct_path(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"cancelled": True, "result_type": "simulated"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            runtime.live_runtime_session_cancel("sess-1", {"request_id": "r1"})
        self.assertEqual(
            seen, [(f"{server.RUNTIME_SESSION_BASE_URL}/sess-1/cancel", "POST")]
        )

    def test_session_summary_uses_get_method(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"result_type": "simulated", "ticks_elapsed": 0})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_runtime_session_summary("sess-1")
        self.assertEqual(
            seen, [(f"{server.RUNTIME_SESSION_BASE_URL}/sess-1/summary", "GET")]
        )
        self.assertEqual(result["ticks_elapsed"], 0)

    def test_session_delete_uses_delete_method(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse({"deleted": True, "result_type": "simulated"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_runtime_session_delete("sess-1")
        self.assertEqual(
            seen, [(f"{server.RUNTIME_SESSION_BASE_URL}/sess-1", "DELETE")]
        )
        self.assertTrue(result["deleted"])

    def test_all_session_methods_unavailable_without_raising(self):
        runtime = server.MiniServingRuntime()
        fallback = {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            self.assertEqual(runtime.live_runtime_session_create({}), fallback)
            self.assertEqual(
                runtime.live_runtime_session_request("sess-1", {"prompt_tokens": 16, "max_output_tokens": 16}),
                fallback,
            )
            self.assertEqual(runtime.live_runtime_session_step("sess-1", {}), fallback)
            self.assertEqual(
                runtime.live_runtime_session_cancel("sess-1", {"request_id": "r1"}), fallback
            )
            self.assertEqual(runtime.live_runtime_session_summary("sess-1"), fallback)
            self.assertEqual(runtime.live_runtime_session_delete("sess-1"), fallback)

    def test_generate_and_qwen_ask_unaffected_by_session_bridge(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            generate_result = runtime.generate({"prompt_tokens": 128, "max_output_tokens": 16})
            ask_result = runtime.ask({"question": "What is the capital of France?"})
        self.assertEqual(generate_result["status"], "completed")
        self.assertIn("answer", ask_result)


class ParseRuntimeSessionPathTest(unittest.TestCase):
    def test_id_and_action(self):
        self.assertEqual(
            server._parse_runtime_session_path("/api/runtime/session/abc/summary"),
            ("abc", "summary"),
        )

    def test_id_only(self):
        self.assertEqual(
            server._parse_runtime_session_path("/api/runtime/session/abc"), ("abc", None)
        )

    def test_bare_session_path_returns_none(self):
        self.assertIsNone(server._parse_runtime_session_path("/api/runtime/session"))

    def test_unrelated_path_returns_none(self):
        self.assertIsNone(server._parse_runtime_session_path("/api/runtime/simulate"))


if __name__ == "__main__":
    unittest.main()
