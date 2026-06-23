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


class CompilerAdapterTest(unittest.TestCase):
    def test_compiled_artifact_passes_through_and_targets_compile_service_url(self):
        runtime = server.MiniServingRuntime()
        fake_payload = {
            "result_type": "compiled_artifact",
            "graph_name": "tiny_gpt_serving",
            "graph_summary": {"model": {"model": "tiny-gpt"}, "op_count": 10},
            "selected_passes": [
                "canonicalize",
                "matmul_bias_relu_fusion",
                "hir_lowering",
                "backend_placement",
                "memory_planning",
            ],
            "fusion_decisions": [
                {"plan_id": "plan_metal", "runtime_action": "dispatch_fused_kernel"}
            ],
            "memory_plan_summary": {"peak_decode_memory_mb": 673},
            "kernel_selection_summary": {"selected_plan_id": "plan_metal"},
            "artifact_paths": ["/tmp/compile_service_x/memory_plan.json"],
            "validation": {"status": "passed", "passed": 26, "failed": 0, "total": 26},
            "truth_boundary": {
                "compiler_logic_executed": True,
                "note": "real pipeline ran",
                "not_claimed": ["no GPU/CUDA/Metal kernel execution occurred"],
            },
            "git_commit": "abc1234",
        }
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append((request.full_url, request.get_method()))
            return FakeResponse(fake_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_compile({"graph_name": "tiny_gpt_serving"})

        self.assertEqual(seen, [(server.COMPILE_SERVICE_URL, "POST")])
        self.assertEqual(result, fake_payload)
        self.assertEqual(result["result_type"], "compiled_artifact")
        self.assertTrue(result["truth_boundary"]["compiler_logic_executed"])

    def test_simulated_compile_passes_through_unchanged(self):
        runtime = server.MiniServingRuntime()
        fake_payload = {
            "result_type": "simulated_compile",
            "graph_name": "not-real",
            "graph_summary": {"source": None, "model": {}, "op_count": 0},
            "selected_passes": [],
            "fusion_decisions": [],
            "memory_plan_summary": {},
            "kernel_selection_summary": {
                "selected_plan_id": None,
                "selection_reason": None,
                "candidates": [],
            },
            "artifact_paths": [],
            "validation": {"status": "not_run", "passed": 0, "failed": 0, "total": 0},
            "truth_boundary": {
                "compiler_logic_executed": False,
                "note": "no real graph definition found for graph_name='not-real'",
                "not_claimed": ["this is not derived from any real graph"],
            },
            "git_commit": "abc1234",
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(fake_payload)):
            result = runtime.live_compile({"graph_name": "not-real"})
        self.assertEqual(result, fake_payload)
        self.assertEqual(result["result_type"], "simulated_compile")
        self.assertEqual(result["artifact_paths"], [])
        self.assertFalse(result["truth_boundary"]["compiler_logic_executed"])

    def test_unavailable_returns_status_unavailable_without_raising(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.live_compile({"graph_name": "tiny_gpt_serving"})
        self.assertEqual(
            result, {"status": "unavailable", "source": "ml-graph-compiler-runtime-http"}
        )

    def test_default_payload_when_none(self):
        runtime = server.MiniServingRuntime()
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append(request.data)
            return FakeResponse({"result_type": "compiled_artifact"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_compile(None)
        self.assertEqual(seen, [b"{}"])
        self.assertEqual(result["result_type"], "compiled_artifact")

    def test_generate_unaffected_by_compiler_adapter(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.generate({"prompt_tokens": 128, "max_output_tokens": 16})
        self.assertEqual(result["status"], "completed")
        self.assertIn("ttft_ms", result)
        self.assertIn("tpot_ms", result)

    def test_qwen_ask_unaffected_by_compiler_adapter(self):
        runtime = server.MiniServingRuntime()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = runtime.ask({"question": "What is the capital of France?"})
        self.assertIn("answer", result)
        self.assertIn("source_status", result)

    def test_runtime_simulate_adapter_unaffected_by_compiler_adapter_addition(self):
        runtime = server.MiniServingRuntime()
        fake_payload = {"result_type": "simulated", "policy": "inflight_paged_kv_continuous_batching"}
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append(request.full_url)
            return FakeResponse(fake_payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = runtime.live_runtime_simulate({"prompt_tokens": 256, "max_output_tokens": 32})
        self.assertEqual(seen, [server.RUNTIME_SIMULATE_URL])
        self.assertEqual(result, fake_payload)

    def test_compiler_adapter_status_and_available_false_when_unreachable(self):
        adapter = server.CompilerAdapter(url="http://127.0.0.1:1/compile", timeout=0.2)
        status = adapter.status()
        self.assertFalse(status["ready"])
        self.assertFalse(adapter.available())


if __name__ == "__main__":
    unittest.main()
