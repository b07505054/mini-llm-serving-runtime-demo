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


if __name__ == "__main__":
    unittest.main()
