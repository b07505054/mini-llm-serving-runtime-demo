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


class FakeQwen:
    def __init__(self, generated_tokens=4):
        self.calls = []
        self.ask_calls = []
        self.stepwise_calls = []
        self.before_decode_states = []
        self.max_new_tokens = 96
        self.model_id = "fake-qwen"
        self.device = "cpu"
        self.device_setting = "cpu"
        self.generated_tokens = generated_tokens

    def ask(self, prompt_contract, policy):
        self.ask_calls.append((prompt_contract, policy))
        self.calls.append((prompt_contract, policy))
        return self._result(policy, self.generated_tokens)

    def ask_stepwise(self, prompt_contract, policy=None, before_decode_step=None):
        policy = policy or {}
        self.stepwise_calls.append((prompt_contract, policy, before_decode_step))
        self.calls.append((prompt_contract, policy))
        max_tokens = min(self.generated_tokens, int(policy.get("max_new_tokens", self.generated_tokens)))
        generated = 0
        stop_reason = "max_new_tokens"
        decode_steps = []
        for index in range(max_tokens):
            state = {
                "step_index": index + 1,
                "request_id": policy.get("request_id"),
                "generated_tokens": generated,
                "max_new_tokens": max_tokens,
                "decode_steps": decode_steps,
            }
            self.before_decode_states.append(state)
            if before_decode_step:
                control = before_decode_step(index + 1, state) or {}
                if control.get("continue") is False:
                    stop_reason = control.get("reason") or "runtime_decode_stopped"
                    break
            generated += 1
            decode_steps.append(
                {
                    "step": index + 1,
                    "token_id": index + 100,
                    "text": f"tok{index + 1}",
                    "latency_ms": 0.5,
                    "cumulative_output": " ".join(f"tok{i + 1}" for i in range(generated)),
                }
            )
        return self._result(policy, generated, stop_reason=stop_reason, decode_steps=decode_steps)

    def _result(self, policy, generated_tokens, stop_reason="max_new_tokens", decode_steps=None):
        decode_steps = decode_steps or []
        answer = " ".join(f"tok{i + 1}" for i in range(generated_tokens)) or "fake answer"
        return {
            "status": "completed",
            "source_status": "qwen_live",
            "model_id": self.model_id,
            "device": self.device,
            "policy": policy,
            "answer": answer,
            "prompt_tokens": 12,
            "generated_tokens": generated_tokens,
            "max_new_tokens": policy.get("max_new_tokens", 4),
            "stop_reason": stop_reason,
            "prefill_ms": 1.0,
            "ttft_ms": 2.0,
            "tpot_ms": 0.5,
            "total_latency_ms": 3.0,
            "tokens_per_second": 10.0,
            "cache_type": "fake_past_key_values",
            "compiler_plan": {"artifact_source": "fake"},
            "runtime_trace": [],
            "decode_steps": decode_steps,
            "live_qwen_metrics": {
                "prompt_tokens": 12,
                "generated_tokens": generated_tokens,
                "total_latency_ms": 3.0,
                "tpot_ms": 0.5,
                "tokens_per_second": 10.0,
                "max_new_tokens": policy.get("max_new_tokens", 4),
            },
        }


class FakeRuntimeSession:
    def __init__(self, admission, create=None, summary=None, step_results=None):
        self.create_response = create or {"session_id": "sess-test", "result_type": "simulated"}
        self.admission = admission
        self.summary = summary or {
            "result_type": "simulated",
            "resident_request_ids": ["ask-test"],
            "finished_request_ids": [],
            "rejected_request_ids": [],
            "requests_submitted": 1,
            "ticks_elapsed": 0,
            "kv_page_lifecycle": {
                "total_pages": 64,
                "page_size_tokens": 16,
                "resident_pages": 2,
                "allocated_pages": 2,
            },
        }
        self.created_payloads = []
        self.request_payloads = []
        self.step_payloads = []
        self.deleted_session_ids = []
        self.step_results = list(step_results or [])

    def create_session(self, payload):
        self.created_payloads.append(payload)
        return self.create_response

    def session_request(self, session_id, payload):
        self.request_payloads.append((session_id, payload))
        return self.admission

    def session_step(self, session_id, payload):
        self.step_payloads.append((session_id, payload))
        if self.step_results:
            return self.step_results.pop(0)
        return {"result_type": "simulated", "events": []}

    def session_summary(self, session_id):
        return self.summary

    def delete_session(self, session_id):
        self.deleted_session_ids.append(session_id)
        return {"deleted": True, "session_id": session_id, "result_type": "simulated"}


class FakeTensor:
    def __init__(self, value=None, shape=(1, 1), dtype="int64", device="cpu"):
        self.value = value
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def item(self):
        return self.value


class FakeLogits:
    def __init__(self, token_id):
        self.token_id = token_id

    def __getitem__(self, _key):
        return self


class FakeTorch:
    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def inference_mode(self):
        return self._InferenceMode()

    def argmax(self, logits, dim=-1, keepdim=True):
        return FakeTensor(logits.token_id, shape=(1, 1))

    def ones(self, shape, device=None, dtype=None):
        return FakeTensor(1, shape=shape, dtype=dtype or "int64", device=device or "cpu")

    def cat(self, tensors, dim=-1):
        left = tensors[0]
        right = tensors[1]
        if dim == -1 and len(left.shape) == 2 and len(right.shape) == 2:
            return FakeTensor(shape=(left.shape[0], left.shape[1] + right.shape[1]), dtype=left.dtype, device=left.device)
        return left


class FakeTokenizer:
    eos_token_id = None

    def __call__(self, prompt_text, return_tensors="pt"):
        return {
            "input_ids": FakeTensor(shape=(1, 2)),
            "attention_mask": FakeTensor(shape=(1, 2)),
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(message["content"] for message in messages)

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class FakeModelOutputs:
    def __init__(self, token_id, past_key_values):
        self.logits = FakeLogits(token_id)
        self.past_key_values = past_key_values


class FakeModel:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = []
        self.next_index = 0
        self.config = type(
            "FakeConfig",
            (),
            {
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 8,
                "vocab_size": 1000,
            },
        )()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        token_id = self.token_ids[min(self.next_index, len(self.token_ids) - 1)]
        self.next_index += 1
        return FakeModelOutputs(token_id, object())

    def parameters(self):
        return iter([])


def make_fake_qwen_adapter(token_ids=(101, 102, 103, 104)):
    adapter = server.QwenRuntimeAdapter.__new__(server.QwenRuntimeAdapter)
    adapter.model_id = "fake-qwen"
    adapter.device_setting = "cpu"
    adapter.max_new_tokens = 96
    adapter.torch = FakeTorch()
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeModel(token_ids)
    adapter.device = "cpu"
    adapter.last_error = None
    adapter.last_status = {"ready": True}
    adapter.status = lambda load_model=True: {"ready": True, "status": "ready"}
    adapter._sync = lambda: None
    return adapter


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

    def test_qwen_runtime_adapter_ask_delegates_to_ask_stepwise(self):
        adapter = server.QwenRuntimeAdapter.__new__(server.QwenRuntimeAdapter)
        seen = []
        expected = {
            "status": "completed",
            "answer": "delegated",
            "decode_steps": [],
            "generated_tokens": 0,
        }

        def fake_ask_stepwise(prompt_contract, policy=None, before_decode_step=None):
            seen.append((prompt_contract, policy, before_decode_step))
            return expected

        adapter.ask_stepwise = fake_ask_stepwise
        result = server.QwenRuntimeAdapter.ask(adapter, {"prompt": "hello"}, {"max_new_tokens": 2})

        self.assertIs(result, expected)
        self.assertEqual(seen, [({"prompt": "hello"}, {"max_new_tokens": 2}, None)])

    def test_qwen_runtime_adapter_stepwise_callback_called_once_per_token(self):
        adapter = make_fake_qwen_adapter((101, 102, 103, 104))
        callback_steps = []

        def before_decode_step(step_index, session_state):
            callback_steps.append((step_index, session_state["generated_tokens"]))
            return {"continue": True}

        result = adapter.ask_stepwise(
            {"question": "token loop", "system_rules": []},
            {"max_new_tokens": 3},
            before_decode_step=before_decode_step,
        )

        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(callback_steps, [(1, 0), (2, 1), (3, 2)])
        self.assertEqual([step["token_id"] for step in result["decode_steps"]], [101, 102, 103])

    def test_qwen_runtime_adapter_stepwise_callback_stop_happens_before_decode(self):
        adapter = make_fake_qwen_adapter((101, 102, 103, 104))

        def before_decode_step(step_index, session_state):
            if step_index == 3:
                return {"continue": False, "reason": "runtime_step_failed"}
            return {"continue": True}

        result = adapter.ask_stepwise(
            {"question": "stop before third token", "system_rules": []},
            {"max_new_tokens": 4},
            before_decode_step=before_decode_step,
        )

        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["stop_reason"], "runtime_step_failed")
        self.assertEqual([step["token_id"] for step in result["decode_steps"]], [101, 102])

    def test_qwen_decode_one_token_advances_exactly_one_token(self):
        adapter = make_fake_qwen_adapter((101, 102, 103))
        session = adapter.prepare_decode_session(
            {"question": "one token", "system_rules": []},
            {"max_new_tokens": 3},
        )

        result = adapter.decode_one_token(session)

        self.assertTrue(result["decoded"])
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(session.generated_ids, [101])
        self.assertEqual(len(session.decode_steps), 1)
        self.assertFalse(session.finished)

    def test_qwen_decode_sessions_are_independent(self):
        adapter = make_fake_qwen_adapter((101, 102, 103, 104))
        first = adapter.prepare_decode_session(
            {"question": "first", "system_rules": []},
            {"max_new_tokens": 2},
        )
        second = adapter.prepare_decode_session(
            {"question": "second", "system_rules": []},
            {"max_new_tokens": 2},
        )

        adapter.decode_one_token(first)

        self.assertEqual(len(first.generated_ids), 1)
        self.assertEqual(second.generated_ids, [])
        self.assertEqual(second.decode_steps, [])
        self.assertIsNot(first.generated_ids, second.generated_ids)
        self.assertIsNot(first.decode_steps, second.decode_steps)

        adapter.decode_one_token(second)

        self.assertEqual(len(first.generated_ids), 1)
        self.assertEqual(len(second.generated_ids), 1)
        self.assertEqual(len(first.decode_steps), 1)
        self.assertEqual(len(second.decode_steps), 1)

    def test_qwen_finalize_decode_session_matches_ask_stepwise_shape(self):
        helper_adapter = make_fake_qwen_adapter((101, 102, 103))
        session = helper_adapter.prepare_decode_session(
            {"question": "shape", "system_rules": []},
            {"max_new_tokens": 2},
        )
        while not session.finished:
            helper_adapter.decode_one_token(session)
        helper_result = helper_adapter.finalize_decode_session(session)

        stepwise_adapter = make_fake_qwen_adapter((101, 102, 103))
        stepwise_result = stepwise_adapter.ask_stepwise(
            {"question": "shape", "system_rules": []},
            {"max_new_tokens": 2},
        )

        expected_keys = {
            "status",
            "source_status",
            "model_id",
            "device",
            "policy",
            "answer",
            "prompt_tokens",
            "generated_tokens",
            "max_new_tokens",
            "prefill_ms",
            "ttft_ms",
            "tpot_ms",
            "total_latency_ms",
            "tokens_per_second",
            "cache_type",
            "stop_reason",
            "compiler_plan",
            "runtime_trace",
            "decode_steps",
            "live_qwen_metrics",
        }
        self.assertEqual(set(helper_result), expected_keys)
        self.assertEqual(set(helper_result), set(stepwise_result))
        self.assertEqual(helper_result["generated_tokens"], stepwise_result["generated_tokens"])
        self.assertEqual(helper_result["stop_reason"], stepwise_result["stop_reason"])

    def test_fake_qwen_stepwise_callback_called_once_per_generated_token(self):
        qwen = FakeQwen(generated_tokens=3)
        callback_steps = []

        def before_decode_step(step_index, session_state):
            callback_steps.append((step_index, session_state["generated_tokens"]))
            return {"continue": True}

        result = qwen.ask_stepwise(
            {"question": "token loop"},
            {"max_new_tokens": 3},
            before_decode_step=before_decode_step,
        )

        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(callback_steps, [(1, 0), (2, 1), (3, 2)])
        self.assertEqual(len(result["decode_steps"]), 3)

    def test_fake_qwen_stepwise_callback_stop_happens_before_decode(self):
        qwen = FakeQwen(generated_tokens=4)

        def before_decode_step(step_index, session_state):
            if step_index == 3:
                return {"continue": False, "reason": "runtime_step_failed"}
            return {"continue": True}

        result = qwen.ask_stepwise(
            {"question": "stop before third token"},
            {"max_new_tokens": 4},
            before_decode_step=before_decode_step,
        )

        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["stop_reason"], "runtime_step_failed")
        self.assertEqual([step["step"] for step in result["decode_steps"]], [1, 2])

    def test_qwen_ask_with_runtime_session_admitted_calls_qwen_and_attaches_ownership(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen()
        fake_session = FakeRuntimeSession(
            admission={
                "request_id": "ask-test",
                "result_type": "simulated",
                "admitted": True,
                "reason": "fits_session_kv_budget",
                "allocated_pages": [0, 1],
            }
        )
        runtime.runtime_simulate = fake_session

        result = runtime.ask(
            {
                "request_id": "ask-test",
                "question": "Explain runtime admission.",
                "prompt_tokens": 16,
                "max_output_tokens": 4,
                "include_runtime_session": True,
            }
        )

        self.assertEqual(len(runtime.qwen.calls), 2)
        self.assertEqual(len(runtime.qwen.ask_calls), 1)
        self.assertEqual(len(runtime.qwen.stepwise_calls), 1)
        self.assertEqual(result["runtime_ownership"]["status"], "completed")
        self.assertTrue(result["runtime_ownership"]["authoritative"])
        self.assertTrue(result["runtime_ownership"]["runtime_owned_decode"])
        self.assertEqual(result["runtime_ownership"]["runtime_steps"], result["qwen"]["generated_tokens"])
        self.assertEqual(result["runtime_ownership"]["steps_run"], result["qwen"]["generated_tokens"])
        self.assertEqual(len(fake_session.step_payloads), result["qwen"]["generated_tokens"])
        self.assertTrue(result["runtime_ownership"]["runtime_qwen_step_alignment"]["matches"])
        self.assertEqual(result["runtime_ownership"]["admission"]["reason"], "fits_session_kv_budget")

    def test_qwen_ask_with_runtime_session_rejected_does_not_call_qwen_or_mutate_simulator_rejects(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen()
        before_rejects = runtime.rejected_requests
        fake_session = FakeRuntimeSession(
            admission={
                "request_id": "ask-test",
                "result_type": "simulated",
                "admitted": False,
                "reason": "insufficient_free_kv_pages",
            },
            summary={
                "result_type": "simulated",
                "resident_request_ids": [],
                "finished_request_ids": [],
                "rejected_request_ids": ["ask-test"],
                "requests_submitted": 1,
                "ticks_elapsed": 0,
                "kv_page_lifecycle": {
                    "total_pages": 2,
                    "page_size_tokens": 16,
                    "resident_pages": 0,
                    "allocated_pages": 0,
                },
            },
        )
        runtime.runtime_simulate = fake_session

        result = runtime.ask(
            {
                "request_id": "ask-test",
                "question": "This should be rejected.",
                "prompt_tokens": 1024,
                "max_output_tokens": 1024,
                "include_runtime_session": True,
            }
        )

        self.assertEqual(runtime.qwen.calls, [])
        self.assertEqual(runtime.qwen.ask_calls, [])
        self.assertEqual(runtime.qwen.stepwise_calls, [])
        self.assertEqual(result["status"], "runtime_rejected")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["source_status"], "runtime_rejected_before_qwen")
        self.assertEqual(result["runtime_ownership"]["status"], "rejected")
        self.assertEqual(result["runtime_ownership"]["reject_count_delta"], 1)
        self.assertEqual(result["runtime_ownership"]["reject_reason"], "insufficient_free_kv_pages")
        self.assertEqual(runtime.rejected_requests, before_rejects)

    def test_qwen_ask_with_runtime_session_unavailable_continues_qwen(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen()
        runtime.runtime_simulate = FakeRuntimeSession(
            admission={},
            create={"status": "unavailable", "source": "heterogeneous-runtime-http"},
        )

        result = runtime.ask(
            {
                "request_id": "ask-test",
                "question": "Continue despite runtime infra failure.",
                "include_runtime_session": True,
            }
        )

        self.assertEqual(len(runtime.qwen.calls), 2)
        self.assertEqual(len(runtime.qwen.ask_calls), 2)
        self.assertEqual(runtime.qwen.stepwise_calls, [])
        self.assertEqual(result["runtime_ownership"]["status"], "unavailable")
        self.assertFalse(result["runtime_ownership"]["runtime_owned_decode"])
        self.assertIn("answer", result)

    def test_qwen_ask_without_runtime_session_makes_no_session_calls(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen()
        fake_session = FakeRuntimeSession(admission={"admitted": True})
        runtime.runtime_simulate = fake_session

        result = runtime.ask({"request_id": "ask-test", "question": "Normal path."})

        self.assertEqual(len(runtime.qwen.calls), 2)
        self.assertNotIn("runtime_ownership", result)
        self.assertEqual(fake_session.created_payloads, [])
        self.assertEqual(fake_session.request_payloads, [])

    def test_runtime_session_max_steps_is_clamped_to_32_and_allows_zero(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen(generated_tokens=64)
        fake_session = FakeRuntimeSession(admission={"admitted": True, "reason": "fits_session_kv_budget"})
        runtime.runtime_simulate = fake_session

        result = runtime.ask(
            {
                "request_id": "ask-test",
                "question": "Clamp steps.",
                "prompt_tokens": 16,
                "max_output_tokens": 64,
                "include_runtime_session": True,
                "runtime_session_max_steps": 100,
            }
        )
        self.assertEqual(result["runtime_ownership"]["steps_run"], 32)
        self.assertEqual(result["qwen"]["generated_tokens"], 32)

        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen()
        fake_session = FakeRuntimeSession(admission={"admitted": True, "reason": "fits_session_kv_budget"})
        runtime.runtime_simulate = fake_session
        result = runtime.ask(
            {
                "request_id": "ask-test-zero",
                "question": "Zero steps.",
                "prompt_tokens": 16,
                "max_output_tokens": 64,
                "include_runtime_session": True,
                "runtime_session_max_steps": -5,
            }
        )
        self.assertEqual(result["runtime_ownership"]["steps_run"], 0)
        self.assertEqual(result["qwen"]["generated_tokens"], 0)
        self.assertEqual(fake_session.step_payloads, [])

    def test_runtime_step_failure_mid_generation_stops_before_next_token(self):
        runtime = server.MiniServingRuntime()
        runtime.qwen = FakeQwen(generated_tokens=4)
        fake_session = FakeRuntimeSession(
            admission={"admitted": True, "reason": "fits_session_kv_budget"},
            step_results=[
                {"result_type": "simulated", "events": []},
                {"result_type": "simulated", "events": []},
                {"error": "runtime_session_step_failed", "events": []},
            ],
        )
        runtime.runtime_simulate = fake_session

        result = runtime.ask(
            {
                "request_id": "ask-test",
                "question": "Fail during decode.",
                "prompt_tokens": 16,
                "max_output_tokens": 4,
                "include_runtime_session": True,
            }
        )

        ownership = result["runtime_ownership"]
        self.assertEqual(ownership["status"], "partial")
        self.assertTrue(ownership["runtime_owned_decode"])
        self.assertEqual(ownership["runtime_steps"], 2)
        self.assertEqual(result["qwen"]["generated_tokens"], 2)
        self.assertEqual(len(fake_session.step_payloads), 3)
        self.assertEqual(ownership["runtime_decode_stop_reason"], "runtime_session_step_failed")
        self.assertEqual(
            ownership["runtime_qwen_step_alignment"],
            {
                "runtime_steps": 2,
                "generated_tokens": 2,
                "matches": True,
                "mismatch_reason": None,
            },
        )


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
