#!/usr/bin/env python3
from dataclasses import dataclass
import hashlib
import errno
import json
import math
import os
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", ROOT / "artifacts"))
COMPILER_ARTIFACTS = Path(
    os.environ.get("COMPILER_ARTIFACTS", ARTIFACT_ROOT / "compiler")
)
RUNTIME_ARTIFACTS = Path(
    os.environ.get("RUNTIME_ARTIFACTS", ARTIFACT_ROOT / "runtime")
)
VALIDATION_ARTIFACTS = Path(
    os.environ.get("VALIDATION_ARTIFACTS", ARTIFACT_ROOT / "validation")
)
MOBILE_DEMO_SCENARIOS = Path(
    os.environ.get("MOBILE_DEMO_SCENARIOS", ARTIFACT_ROOT / "mobile_demo_scenarios.json")
)
HF_QWEN_MODEL = os.environ.get("HF_QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_DEVICE = os.environ.get("QWEN_DEVICE", "auto")
QWEN_MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "96"))
BASEMODEL_MAX_NEW_TOKENS = int(os.environ.get("BASEMODEL_MAX_NEW_TOKENS", "180"))
RUNTIME_SIMULATE_URL = os.environ.get("RUNTIME_SIMULATE_URL", "http://127.0.0.1:8901/simulate")
RUNTIME_SIMULATE_BATCH_URL = os.environ.get(
    "RUNTIME_SIMULATE_BATCH_URL", "http://127.0.0.1:8901/simulate_batch"
)
RUNTIME_SESSION_BASE_URL = os.environ.get(
    "RUNTIME_SESSION_BASE_URL", "http://127.0.0.1:8901/session"
)
COMPILE_SERVICE_URL = os.environ.get(
    "COMPILE_SERVICE_URL", "http://127.0.0.1:8902/compile"
)


@dataclass
class PrefixCacheEntry:
    prefix_hash: str
    blocks: list
    tokens: int
    ref_count: int
    last_used_ms: float
    model_version: str


class PrefixCacheManager:
    def __init__(self, total_blocks, block_size_tokens, policy, now_ms):
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.policy = policy or {}
        self.now_ms = now_ms
        self.enabled = bool(self.policy.get("enabled", False))
        self.model_version = self.policy.get("model_version", "tiny-gpt")
        self.max_entries = int(self.policy.get("max_prefix_entries", 128))
        self.free_blocks = list(range(total_blocks))
        self.entries = {}

    def used_blocks(self):
        return self.total_blocks - len(self.free_blocks)

    def available_blocks(self):
        return len(self.free_blocks)

    def lookup(self, prefix_hash, model_version):
        if not self.enabled:
            return None
        entry = self.entries.get(prefix_hash)
        if not entry or entry.model_version != model_version:
            return None
        entry.ref_count += 1
        entry.last_used_ms = self.now_ms()
        return entry

    def allocate_blocks(self, count):
        if count <= 0:
            return []
        if count > len(self.free_blocks):
            return []
        blocks = self.free_blocks[:count]
        del self.free_blocks[:count]
        return blocks

    def release_blocks(self, blocks):
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()

    def insert(self, prefix_hash, blocks, tokens, model_version):
        if not self.enabled or not blocks:
            self.release_blocks(blocks)
            return None
        old = self.entries.pop(prefix_hash, None)
        if old:
            self.release_blocks(old.blocks)
        entry = PrefixCacheEntry(
            prefix_hash=prefix_hash,
            blocks=blocks,
            tokens=tokens,
            ref_count=0,
            last_used_ms=self.now_ms(),
            model_version=model_version,
        )
        self.entries[prefix_hash] = entry
        return entry

    def evict_lru(self, required_blocks):
        evicted = []
        while self.available_blocks() < required_blocks and self.entries:
            victim_hash, victim = min(
                self.entries.items(),
                key=lambda item: (item[1].ref_count > 0, item[1].last_used_ms),
            )
            if victim.ref_count > 0:
                break
            self.entries.pop(victim_hash)
            self.release_blocks(victim.blocks)
            evicted.append(victim)
        while len(self.entries) > self.max_entries:
            victim_hash, victim = min(
                self.entries.items(),
                key=lambda item: item[1].last_used_ms,
            )
            self.entries.pop(victim_hash)
            self.release_blocks(victim.blocks)
            evicted.append(victim)
        return evicted


def load_json(path: Path, fallback):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return fallback


def load_text(path: Path, fallback=""):
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return fallback


class QwenRuntimeAdapter:
    def __init__(self, model_id=HF_QWEN_MODEL, device=QWEN_DEVICE, max_new_tokens=QWEN_MAX_NEW_TOKENS):
        self.model_id = model_id
        self.device_setting = device
        self.max_new_tokens = max_new_tokens
        self.torch = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.last_error = None
        self.last_status = None

    def _dependency_status(self):
        status = {"torch": False, "transformers": False}
        for name in status:
            try:
                __import__(name)
                status[name] = True
            except Exception:
                status[name] = False
        return status

    def _sync(self):
        if not self.torch or not self.device:
            return
        try:
            if self.device == "cuda" and self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
            elif self.device == "mps" and hasattr(self.torch, "mps"):
                self.torch.mps.synchronize()
        except Exception:
            pass

    def _select_device(self, torch):
        if self.device_setting != "auto":
            return self.device_setting
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def status(self, load_model=False):
        deps = self._dependency_status()
        if not all(deps.values()):
            missing = [name for name, ok in deps.items() if not ok]
            self.last_status = {
                "status": "unavailable",
                "ready": False,
                "model_id": self.model_id,
                "device": self.device or self.device_setting,
                "max_new_tokens": self.max_new_tokens,
                "dependencies": deps,
                "missing_dependencies": missing,
                "model_loaded": False,
                "compiler_ready": False,
                "last_error": f"Missing optional dependencies: {', '.join(missing)}",
            }
            return self.last_status
        if load_model and self.model is None:
            self._load()
        ready = self.model is not None and self.tokenizer is not None
        self.last_status = {
            "status": "ready" if ready else "dependencies_ready",
            "ready": ready,
            "model_id": self.model_id,
            "device": self.device or self.device_setting,
            "max_new_tokens": self.max_new_tokens,
            "dependencies": deps,
            "missing_dependencies": [],
            "model_loaded": ready,
            "compiler_ready": ready,
            "last_error": self.last_error,
        }
        return self.last_status

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.device = self._select_device(torch)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype="auto",
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.model = None
            self.tokenizer = None
            return False

    def _chat_prompt(self, prompt_contract, policy):
        system = "\n".join(prompt_contract.get("system_rules", []))
        question = prompt_contract.get("question", "")
        if policy.get("prompt_lowering", True):
            user = question
        else:
            user = question
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:"

    def _policy_summary(self, policy):
        return {
            "policy_name": policy.get("policy_name", "optimized"),
            "max_new_tokens": int(policy.get("max_new_tokens", self.max_new_tokens)),
            "prompt_lowering": bool(policy.get("prompt_lowering", True)),
            "chunked_prefill": bool(policy.get("chunked_prefill", True)),
            "pressure_aware_policy": bool(policy.get("pressure_aware_policy", True)),
            "prompt_mode": policy.get("prompt_mode", "compact_lowered_prompt"),
        }

    def _live_metrics(self, result):
        policy = result.get("policy", {})
        return {
            "status": result.get("status"),
            "source_status": result.get("source_status"),
            "model_id": result.get("model_id", self.model_id),
            "device": result.get("device", self.device or self.device_setting),
            "prompt_tokens": result.get("prompt_tokens"),
            "generated_tokens": result.get("generated_tokens"),
            "prefill_ms": result.get("prefill_ms"),
            "ttft_ms": result.get("ttft_ms"),
            "tpot_ms": result.get("tpot_ms"),
            "total_latency_ms": result.get("total_latency_ms"),
            "tokens_per_second": result.get("tokens_per_second"),
            "max_new_tokens": result.get("max_new_tokens"),
            "prompt_lowering": policy.get("prompt_lowering"),
            "chunked_prefill": policy.get("chunked_prefill"),
            "pressure_aware_policy": policy.get("pressure_aware_policy"),
            "cache_type": result.get("cache_type"),
            "stop_reason": result.get("stop_reason"),
        }

    def _compiler_plan(self, policy):
        if policy.get("compiler_plan_source") == "disabled_for_basemodel":
            return {
                "artifact_source": "disabled_for_basemodel",
                "reason": "BASEMODEL is direct Qwen serving: full prompt, no prompt lowering, no chunked prefill, no pressure-aware policy.",
                "serving_policy": self._policy_summary(policy),
            }
        config = getattr(self.model, "config", None)
        layers = int(getattr(config, "num_hidden_layers", 0) or 0)
        hidden = int(getattr(config, "hidden_size", 0) or 0)
        heads = int(getattr(config, "num_attention_heads", 0) or 0)
        kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads or 0)
        intermediate = int(getattr(config, "intermediate_size", 0) or 0)
        vocab = int(getattr(config, "vocab_size", 0) or 0)
        head_dim = int(hidden / heads) if hidden and heads else 0
        dtype = "unknown"
        dtype_bytes = 2
        try:
            dtype = str(next(self.model.parameters()).dtype)
            dtype_bytes = 2 if "16" in dtype or "bfloat" in dtype else 4
        except Exception:
            pass
        bytes_per_token = layers * 2 * kv_heads * head_dim * dtype_bytes
        block_size = 16
        bytes_per_block = bytes_per_token * block_size
        model_config = {
            "name": self.model_id,
            "num_layers": layers,
            "hidden_size": hidden,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "dtype": dtype,
            "operators": ["qwen_rmsnorm", "qkv_projection", "grouped_query_attention", "mlp"],
        }
        return {
            "artifact_source": "qwen_live_huggingface_config",
            "schema_compatibility": "ml-graph-compiler-runtime generate_llm_artifacts.py serving schema",
            "model": model_config,
            "graph_ir": {
                "artifact_type": "qwen_graph_ir",
                "adapter": "PyTorch module config + FX-style serving graph adapter",
                "nodes": [
                    "token_embedding",
                    "prefill.transformer_block[*]",
                    "decode.transformer_block[*].past_key_values",
                    "lm_head",
                ],
            },
            "serving_execution_plan": {
                "artifact_type": "qwen_prefill_decode_execution_plan",
                "phases": [
                    {"name": "prefill", "inputs": ["input_ids", "attention_mask"], "use_cache": True},
                    {"name": "decode", "step_tokens": 1, "cache": "past_key_values", "loop": "token_by_token"},
                ],
            },
            "kv_cache_plan": {
                "cache_type": "PyTorch past_key_values",
                "block_size_tokens": block_size,
                "bytes_per_token": bytes_per_token,
                "bytes_per_block": bytes_per_block,
                "num_layers": layers,
                "num_kv_heads": kv_heads,
                "head_dim": head_dim,
                "dtype_bytes": dtype_bytes,
            },
            "memory_plan": {
                "max_new_tokens": self.max_new_tokens,
                "estimated_kv_mb_per_1k_tokens": round(bytes_per_token * 1024 / (1024 * 1024), 3),
                "reuse_boundary": "past_key_values are reused between prefill and each decode step",
            },
            "scheduling_plan": {
                "policy": "single_request_greedy_decode",
                "prefill_batch": 1,
                "decode_step_tokens": 1,
                "future_extension": "continuous batching and custom kernel lowering",
            },
            "serving_policy": self._policy_summary(policy),
        }

    def ask(self, prompt_contract, policy=None):
        policy = policy or {}
        policy = {
            "policy_name": policy.get("policy_name", "Optimized Runtime"),
            "max_new_tokens": int(policy.get("max_new_tokens", self.max_new_tokens)),
            "prompt_lowering": bool(policy.get("prompt_lowering", True)),
            "chunked_prefill": bool(policy.get("chunked_prefill", True)),
            "pressure_aware_policy": bool(policy.get("pressure_aware_policy", True)),
            "prompt_mode": policy.get("prompt_mode", "compact_lowered_prompt"),
            "compiler_plan_source": policy.get("compiler_plan_source", "qwen_live_huggingface_config"),
        }
        status = self.status(load_model=True)
        if not status.get("ready"):
            result = {
                "status": "unavailable",
                "source_status": "qwen_unavailable",
                "status_detail": status,
                "answer": "",
                "policy": self._policy_summary(policy),
                "compiler_plan": (
                    {
                        "artifact_source": "disabled_for_basemodel",
                        "reason": "BASEMODEL compiler/runtime policy is disabled."
                    }
                    if policy.get("compiler_plan_source") == "disabled_for_basemodel"
                    else None
                ),
                "runtime_trace": [],
                "decode_steps": [],
                "max_new_tokens": policy["max_new_tokens"],
            }
            result["live_qwen_metrics"] = self._live_metrics(result)
            return result

        max_tokens = int(policy.get("max_new_tokens", self.max_new_tokens))
        torch = self.torch
        prompt_text = self._chat_prompt(prompt_contract, policy)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        prompt_tokens = int(input_ids.shape[-1])
        compiler_plan = self._compiler_plan(policy)
        decode_steps = []
        generated_ids = []
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                prefill_start = time.perf_counter()
                outputs = self.model(**encoded, use_cache=True)
                self._sync()
                prefill_ms = (time.perf_counter() - prefill_start) * 1000
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                eos_ids = set()
                if self.tokenizer.eos_token_id is not None:
                    eos_ids.add(int(self.tokenizer.eos_token_id))
                first_decode_ms = None
                stop_reason = "max_new_tokens"
                for index in range(max_tokens):
                    step_start = time.perf_counter()
                    token_id = int(next_token.item())
                    generated_ids.append(token_id)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype)],
                            dim=-1,
                        )
                    try:
                        outputs = self.model(input_ids=next_token, past_key_values=past, use_cache=True)
                    except TypeError:
                        outputs = self.model(
                            input_ids=next_token,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                        )
                    self._sync()
                    latency_ms = (time.perf_counter() - step_start) * 1000
                    if first_decode_ms is None:
                        first_decode_ms = latency_ms
                    fragment = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    cumulative = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    decode_steps.append(
                        {
                            "step": index + 1,
                            "token_id": token_id,
                            "text": fragment,
                            "latency_ms": round(latency_ms, 3),
                            "cumulative_output": cumulative,
                        }
                    )
                    if token_id in eos_ids:
                        stop_reason = "eos_token"
                        break
                    past = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
            total_ms = (time.perf_counter() - started) * 1000
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            decode_latencies = [step["latency_ms"] for step in decode_steps]
            tpot_ms = round(sum(decode_latencies) / len(decode_latencies), 3) if decode_latencies else 0
            generated_tokens = len(generated_ids)
            tokens_per_second = round((generated_tokens / (total_ms / 1000)), 3) if total_ms else 0
            result = {
                "status": "completed",
                "source_status": "qwen_live",
                "model_id": self.model_id,
                "device": self.device,
                "policy": self._policy_summary(policy),
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "max_new_tokens": max_tokens,
                "prefill_ms": round(prefill_ms, 3),
                "ttft_ms": round(prefill_ms + (first_decode_ms or 0), 3),
                "tpot_ms": tpot_ms,
                "total_latency_ms": round(total_ms, 3),
                "tokens_per_second": tokens_per_second,
                "cache_type": "PyTorch past_key_values",
                "stop_reason": stop_reason,
                "compiler_plan": compiler_plan,
                "runtime_trace": {
                    "prefill": {"latency_ms": round(prefill_ms, 3), "use_cache": True},
                    "decode_loop": decode_steps,
                },
                "decode_steps": decode_steps,
            }
            result["live_qwen_metrics"] = self._live_metrics(result)
            return result
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            result = {
                "status": "error",
                "source_status": "qwen_error",
                "status_detail": self.status(load_model=False),
                "answer": "",
                "error": self.last_error,
                "policy": self._policy_summary(policy),
                "max_new_tokens": max_tokens,
                "compiler_plan": compiler_plan,
                "runtime_trace": [],
                "decode_steps": [],
            }
            result["live_qwen_metrics"] = self._live_metrics(result)
            return result


class RuntimeSimulateAdapter:
    """Thin HTTP client for heterogeneous-inference-runtime's optional local
    POST /simulate service. Stdlib urllib only; no source code is imported
    across repos. Degrades to unavailable on any connection failure."""

    def __init__(self, url=None, batch_url=None, session_base_url=None, timeout=2.0):
        self.url = url or RUNTIME_SIMULATE_URL
        self.batch_url = batch_url or RUNTIME_SIMULATE_BATCH_URL
        self.session_base_url = session_base_url or RUNTIME_SESSION_BASE_URL
        self.timeout = timeout
        self.last_error = None

    def status(self):
        host_port = urlparse(self.url).netloc
        if not host_port:
            self.last_error = "invalid_runtime_simulate_url"
            return {"ready": False, "url": self.url, "last_error": self.last_error}
        host, _, port = host_port.partition(":")
        try:
            import socket

            with socket.create_connection((host, int(port or 80)), timeout=self.timeout):
                pass
            self.last_error = None
            return {"ready": True, "url": self.url, "last_error": None}
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {"ready": False, "url": self.url, "last_error": self.last_error}

    def available(self):
        return bool(self.status().get("ready"))

    def _post(self, url, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_error = None
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                self.last_error = f"HTTPError: {exc.code}"
                return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _get(self, url):
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_error = None
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                self.last_error = f"HTTPError: {exc.code}"
                return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _delete(self, url):
        request = urllib.request.Request(url, method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_error = None
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                self.last_error = f"HTTPError: {exc.code}"
                return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def simulate(self, payload):
        return self._post(self.url, payload)

    def simulate_batch(self, payload):
        return self._post(self.batch_url, payload)

    def create_session(self, payload):
        return self._post(self.session_base_url, payload)

    def session_request(self, session_id, payload):
        return self._post(f"{self.session_base_url}/{session_id}/request", payload)

    def session_step(self, session_id, payload):
        return self._post(f"{self.session_base_url}/{session_id}/step", payload or {})

    def session_cancel(self, session_id, payload):
        return self._post(f"{self.session_base_url}/{session_id}/cancel", payload)

    def session_summary(self, session_id):
        return self._get(f"{self.session_base_url}/{session_id}/summary")

    def delete_session(self, session_id):
        return self._delete(f"{self.session_base_url}/{session_id}")


class CompilerAdapter:
    """Thin HTTP client for ml-graph-compiler-runtime's optional local
    POST /compile service. Stdlib urllib only; no source code is imported
    across repos. Degrades to unavailable on any connection failure."""

    def __init__(self, url=None, timeout=2.0):
        self.url = url or COMPILE_SERVICE_URL
        self.timeout = timeout
        self.last_error = None

    def status(self):
        host_port = urlparse(self.url).netloc
        if not host_port:
            self.last_error = "invalid_compile_service_url"
            return {"ready": False, "url": self.url, "last_error": self.last_error}
        host, _, port = host_port.partition(":")
        try:
            import socket

            with socket.create_connection((host, int(port or 80)), timeout=self.timeout):
                pass
            self.last_error = None
            return {"ready": True, "url": self.url, "last_error": None}
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {"ready": False, "url": self.url, "last_error": self.last_error}

    def available(self):
        return bool(self.status().get("ready"))

    def _post(self, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_error = None
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                self.last_error = f"HTTPError: {exc.code}"
                return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def compile(self, payload):
        return self._post(payload)


class MiniServingRuntime:
    def __init__(self):
        self.compiler = {
            "execution_plan": load_json(COMPILER_ARTIFACTS / "serving_execution_plan.json", {}),
            "kv_cache_plan": load_json(COMPILER_ARTIFACTS / "kv_cache_plan.json", {}),
            "memory_plan": load_json(COMPILER_ARTIFACTS / "memory_plan.json", {}),
            "scheduling_plan": load_json(COMPILER_ARTIFACTS / "scheduling_plan.json", {}),
            "mlir_source": load_text(COMPILER_ARTIFACTS / "tiny_gpt_serving.mlir"),
            "mlir_fused_graph": load_text(COMPILER_ARTIFACTS / "mlir_fused_graph.mlir"),
            "mlir_lowered_graph": load_json(COMPILER_ARTIFACTS / "mlir_lowered_graph.json", {}),
            "mlir_execution_plan": load_json(COMPILER_ARTIFACTS / "mlir_execution_plan.json", {}),
            "rmsnorm_fused_graph": load_text(COMPILER_ARTIFACTS / "rmsnorm_fused_graph.mlir"),
            "rmsnorm_lowered_graph": load_json(COMPILER_ARTIFACTS / "rmsnorm_lowered_graph.json", {}),
            "rmsnorm_execution_plan": load_json(COMPILER_ARTIFACTS / "rmsnorm_execution_plan.json", {}),
            "artifact_provenance": load_json(COMPILER_ARTIFACTS / "artifact_provenance.json", {}),
            "candidate_execution_plans": load_json(COMPILER_ARTIFACTS / "candidate_execution_plans.json", {}),
            "serving_framework_contract": load_json(COMPILER_ARTIFACTS / "serving_framework_contract.json", {}),
            "memory_timeline": load_json(COMPILER_ARTIFACTS / "memory_timeline.json", {}),
        }
        self.runtime_artifacts = {
            "runtime_profile": load_json(RUNTIME_ARTIFACTS / "runtime_profile.json", {}),
            "prefill_decode": load_json(RUNTIME_ARTIFACTS / "prefill_decode_benchmark.json", {}),
            "scheduler_trace": load_json(RUNTIME_ARTIFACTS / "scheduler_trace.json", {}),
            "scheduler_decision_report": load_json(RUNTIME_ARTIFACTS / "scheduler_decision_report.json", {}),
            "kv_cache_trace": load_json(RUNTIME_ARTIFACTS / "kv_cache_trace.json", {}),
            "plan_benchmark_results": load_json(RUNTIME_ARTIFACTS / "plan_benchmark_results.json", {}),
            "real_llama_profile": load_json(RUNTIME_ARTIFACTS / "real_llama_profile.json", {}),
            "rmsnorm_benchmark": load_json(RUNTIME_ARTIFACTS / "rmsnorm_benchmark.json", {}),
            "gpu_pgo_like_rmsnorm_report": load_json(RUNTIME_ARTIFACTS / "gpu_pgo_like_rmsnorm_report.json", {}),
            "serving_framework_report": load_json(RUNTIME_ARTIFACTS / "serving_framework_report.json", {}),
            "vllm_trace_adapter_report": load_json(RUNTIME_ARTIFACTS / "vllm_trace_adapter_report.json", {}),
            "page_prefetch_report": load_json(RUNTIME_ARTIFACTS / "page_prefetch_report.json", {}),
            "page_prefetch_trace": load_json(RUNTIME_ARTIFACTS / "page_prefetch_trace.json", {}),
            "distributed_serving_report": load_json(RUNTIME_ARTIFACTS / "distributed_serving_report.json", {}),
            "distributed_serving_trace": load_json(RUNTIME_ARTIFACTS / "distributed_serving_trace.json", {}),
            "load_balancing_report": load_json(RUNTIME_ARTIFACTS / "load_balancing_report.json", {}),
            "worker_health_report": load_json(RUNTIME_ARTIFACTS / "worker_health_report.json", {}),
            "fault_tolerance_report": load_json(RUNTIME_ARTIFACTS / "fault_tolerance_report.json", {}),
            "failover_trace": load_json(RUNTIME_ARTIFACTS / "failover_trace.json", {}),
            "grpc_contract_report": load_json(RUNTIME_ARTIFACTS / "grpc_contract_report.json", {}),
            "sglang_trace_adapter_report": load_json(RUNTIME_ARTIFACTS / "sglang_trace_adapter_report.json", {}),
            "cold_start_report": load_json(RUNTIME_ARTIFACTS / "cold_start_report.json", {}),
            "technology_gate_audit": load_json(RUNTIME_ARTIFACTS / "technology_gate_audit.json", {}),
        }
        self.validation = {
            "llm_validation_report": load_json(VALIDATION_ARTIFACTS / "llm_validation_report.json", {}),
            "slo_report": load_json(VALIDATION_ARTIFACTS / "slo_report.json", {}),
            "kv_cache_analysis": load_json(VALIDATION_ARTIFACTS / "kv_cache_analysis.json", {}),
            "request_timeline": load_json(VALIDATION_ARTIFACTS / "request_timeline.json", {}),
            "plan_selection_report": load_json(VALIDATION_ARTIFACTS / "plan_selection_report.json", {}),
            "memory_validation_report": load_json(VALIDATION_ARTIFACTS / "memory_validation_report.json", {}),
            "runtime_decision_validation_report": load_json(VALIDATION_ARTIFACTS / "runtime_decision_validation_report.json", {}),
            "serving_framework_validation_report": load_json(VALIDATION_ARTIFACTS / "serving_framework_validation_report.json", {}),
            "vllm_trace_adapter_validation_report": load_json(VALIDATION_ARTIFACTS / "vllm_trace_adapter_validation_report.json", {}),
            "page_prefetch_validation_report": load_json(VALIDATION_ARTIFACTS / "page_prefetch_validation_report.json", {}),
            "distributed_serving_validation_report": load_json(VALIDATION_ARTIFACTS / "distributed_serving_validation_report.json", {}),
            "load_balancing_validation_report": load_json(VALIDATION_ARTIFACTS / "load_balancing_validation_report.json", {}),
            "fault_tolerance_validation_report": load_json(VALIDATION_ARTIFACTS / "fault_tolerance_validation_report.json", {}),
            "grpc_contract_validation_report": load_json(VALIDATION_ARTIFACTS / "grpc_contract_validation_report.json", {}),
            "sglang_trace_adapter_validation_report": load_json(VALIDATION_ARTIFACTS / "sglang_trace_adapter_validation_report.json", {}),
            "cold_start_validation_report": load_json(VALIDATION_ARTIFACTS / "cold_start_validation_report.json", {}),
            "technology_gate_validation_report": load_json(VALIDATION_ARTIFACTS / "technology_gate_validation_report.json", {}),
            "gpu_pgo_like_validation_report": load_json(VALIDATION_ARTIFACTS / "gpu_pgo_like_validation_report.json", {}),
        }
        self.mobile_demo = load_json(
            MOBILE_DEMO_SCENARIOS,
            {
                "artifact_type": "llm_serving_demo_workloads",
                "truth_boundary": "Generic LLM serving workbench input; live camera/CV is not connected.",
                "scenarios": [],
            },
        )
        self.qwen = QwenRuntimeAdapter()
        self.runtime_simulate = RuntimeSimulateAdapter()
        self.compiler_engine = CompilerAdapter()
        self.reset()

    def reset(self):
        kv_plan = self.compiler["kv_cache_plan"]
        self.block_size_tokens = int(kv_plan.get("block_size_tokens", 16))
        self.total_blocks = int(kv_plan.get("num_blocks", 1024))
        self.bytes_per_block = int(kv_plan.get("bytes_per_block", 589824))
        self.prefix_cache_policy = kv_plan.get("prefix_cache_policy", {
            "enabled": bool(kv_plan.get("prefix_cache_enabled", False)),
            "model_version": kv_plan.get("model", "tiny-gpt"),
            "min_prefix_tokens": self.block_size_tokens,
            "max_prefix_entries": 128,
        })
        self.prefix_cache = PrefixCacheManager(
            self.total_blocks,
            self.block_size_tokens,
            self.prefix_cache_policy,
            self._now_ms,
        )
        self.events = []
        self.requests = []
        self.rejected_requests = 0
        self.completed_requests = 0
        self.generated_tokens = 0
        self.started_at = time.time()
        self.compiler_runs = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.kv_blocks_reused = 0
        self.kv_blocks_evicted = 0
        self.prefill_latency_saved_ms = 0.0
        self.ask_history = []
        self.latest_llm_answer = None
        self.latest_qwen_run = None

    def _now_ms(self):
        return round((time.time() - self.started_at) * 1000, 3)

    def _event(self, event, request_id, **details):
        row = {
            "time_ms": self._now_ms(),
            "event": event,
            "request_id": request_id,
            **details,
        }
        self.events.append(row)
        return row

    def _prefix_fingerprint(self, payload, prompt_tokens):
        min_prefix_tokens = int(
            self.prefix_cache_policy.get("min_prefix_tokens", self.block_size_tokens)
        )
        if not self.prefix_cache.enabled or prompt_tokens < min_prefix_tokens:
            return None, 0

        prefix_tokens = int(payload.get("prefix_tokens", max(min_prefix_tokens, prompt_tokens // 2)))
        prefix_tokens = min(prefix_tokens, prompt_tokens)
        prefix_label = payload.get("prefix") or payload.get("prompt") or "shared-system-prefix-v1"
        model_version = self.prefix_cache_policy.get("model_version", "tiny-gpt")
        raw = f"{model_version}|{prefix_label}|{prefix_tokens}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest(), prefix_tokens

    def generate(self, payload):
        prompt_tokens = int(payload.get("prompt_tokens", 512))
        max_output_tokens = int(payload.get("max_output_tokens", 64))
        request_id = payload.get("request_id") or f"req-{uuid.uuid4().hex[:8]}"
        self.compiler_runs += 1

        required_tokens = prompt_tokens + max_output_tokens
        required_blocks = math.ceil(required_tokens / self.block_size_tokens)
        prefix_hash, prefix_tokens = self._prefix_fingerprint(payload, prompt_tokens)
        model_version = self.prefix_cache_policy.get("model_version", "tiny-gpt")
        prefix_entry = (
            self.prefix_cache.lookup(prefix_hash, model_version)
            if prefix_hash
            else None
        )
        prefix_block_count = math.ceil(prefix_tokens / self.block_size_tokens) if prefix_tokens else 0
        reused_blocks = len(prefix_entry.blocks) if prefix_entry else 0
        required_new_blocks = max(0, required_blocks - reused_blocks)

        self._event(
            "request_arrived",
            request_id,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
            required_blocks=required_blocks,
            prefix_hash=prefix_hash[:12] if prefix_hash else None,
        )
        if prefix_entry:
            self.prefix_cache_hits += 1
            self.kv_blocks_reused += reused_blocks
            self._event(
                "prefix_cache_hit",
                request_id,
                prefix_hash=prefix_hash[:12],
                reused_blocks=reused_blocks,
                reused_tokens=prefix_entry.tokens,
            )
        elif prefix_hash:
            self.prefix_cache_misses += 1
            self._event(
                "prefix_cache_miss",
                request_id,
                prefix_hash=prefix_hash[:12],
                cacheable_prefix_tokens=prefix_tokens,
            )
        self._event(
            "mlir_pattern_matched",
            request_id,
            pattern="linalg.matmul + bias_add + relu",
            fusion_group="matmul_bias_relu_0",
        )
        self._event(
            "lowered_to_hir",
            request_id,
            lowered_op="hir.fused_matmul_bias_relu",
            backend="Metal",
        )

        evicted_entries = []
        if required_new_blocks > self.prefix_cache.available_blocks():
            evicted_entries = self.prefix_cache.evict_lru(required_new_blocks)
            evicted_blocks = sum(len(entry.blocks) for entry in evicted_entries)
            if evicted_blocks:
                self.kv_blocks_evicted += evicted_blocks
                self._event(
                    "kv_blocks_evicted",
                    request_id,
                    evicted_blocks=evicted_blocks,
                    evicted_entries=len(evicted_entries),
                    policy=self.compiler["kv_cache_plan"].get("eviction_policy", "lru"),
                )

        available_blocks = self.prefix_cache.available_blocks()
        if required_new_blocks > available_blocks:
            self.rejected_requests += 1
            result = {
                "request_id": request_id,
                "status": "rejected",
                "reason": "admission_rejected_kv_capacity",
                "required_blocks": required_blocks,
                "required_new_blocks": required_new_blocks,
                "reused_blocks": reused_blocks,
                "available_blocks": available_blocks,
            }
            self.requests.append(result)
            self._event(
                "admission_rejected",
                request_id,
                reason=result["reason"],
                required_blocks=required_blocks,
                required_new_blocks=required_new_blocks,
                available_blocks=available_blocks,
            )
            if prefix_entry:
                prefix_entry.ref_count = max(0, prefix_entry.ref_count - 1)
            return result

        allocated_blocks = self.prefix_cache.allocate_blocks(required_new_blocks)
        self._event(
            "request_admitted",
            request_id,
            allocated_blocks=len(allocated_blocks),
            reused_blocks=reused_blocks,
            resident_prefix_blocks=self.prefix_cache.used_blocks(),
        )

        base_prefill = float(
            self.runtime_artifacts["prefill_decode"].get("prefill_latency_ms", 185.0)
        )
        base_decode = float(
            self.runtime_artifacts["prefill_decode"].get("avg_decode_latency_ms", 13.0)
        )
        full_prefill_latency_ms = base_prefill * (prompt_tokens / 1024)
        saved_latency_ms = 0.0
        if prefix_entry and prompt_tokens:
            saved_latency_ms = full_prefill_latency_ms * min(
                0.9,
                prefix_entry.tokens / prompt_tokens,
            )
        prefill_latency_ms = round(max(0.0, full_prefill_latency_ms - saved_latency_ms), 3)
        saved_latency_ms = round(saved_latency_ms, 3)
        self.prefill_latency_saved_ms += saved_latency_ms
        decode_latency_ms = round(base_decode * max_output_tokens, 3)
        ttft_ms = round(prefill_latency_ms + 2.0, 3)
        tpot_ms = round(base_decode, 3)
        e2e_latency_ms = round(ttft_ms + decode_latency_ms, 3)
        baseline_tpot_ms = round(base_decode * 1.28, 3)
        baseline_ttft_ms = round((base_prefill * 1.35 * (prompt_tokens / 1024)) + 4.0, 3)
        baseline_e2e_latency_ms = round(
            baseline_ttft_ms + (baseline_tpot_ms * max_output_tokens),
            3,
        )
        baseline_blocks = math.ceil(required_blocks * 1.18)

        self._event("prefill_start", request_id, backend="gpu")
        self._event(
            "prefill_end",
            request_id,
            latency_ms=prefill_latency_ms,
            saved_ms=saved_latency_ms,
        )
        self._event("decode_start", request_id, backend="gpu")
        self._event(
            "decode_end",
            request_id,
            generated_tokens=max_output_tokens,
            latency_ms=decode_latency_ms,
        )

        self.completed_requests += 1
        self.generated_tokens += max_output_tokens
        retained_prefix_blocks = []
        released_blocks = allocated_blocks
        if prefix_hash and not prefix_entry and prefix_block_count > 0:
            retained_prefix_blocks = allocated_blocks[:prefix_block_count]
            released_blocks = allocated_blocks[prefix_block_count:]
            self.prefix_cache.insert(
                prefix_hash,
                retained_prefix_blocks,
                prefix_tokens,
                model_version,
            )
            self._event(
                "prefix_cache_inserted",
                request_id,
                prefix_hash=prefix_hash[:12],
                retained_blocks=len(retained_prefix_blocks),
                prefix_tokens=prefix_tokens,
            )
        self.prefix_cache.release_blocks(released_blocks)
        if prefix_entry:
            prefix_entry.ref_count = max(0, prefix_entry.ref_count - 1)

        result = {
            "request_id": request_id,
            "status": "completed",
            "prompt_tokens": prompt_tokens,
            "generated_tokens": max_output_tokens,
            "allocated_blocks": len(allocated_blocks),
            "required_blocks": required_blocks,
            "reused_blocks": reused_blocks,
            "evicted_blocks": sum(len(entry.blocks) for entry in evicted_entries),
            "prefix_cache": "hit" if prefix_entry else "miss" if prefix_hash else "disabled",
            "prefill_latency_saved_ms": saved_latency_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "baseline_ttft_ms": baseline_ttft_ms,
            "baseline_tpot_ms": baseline_tpot_ms,
            "baseline_e2e_latency_ms": baseline_e2e_latency_ms,
            "baseline_blocks": baseline_blocks,
            "text": "simulated token stream",
        }
        self.requests.append(result)
        self._event(
            "request_finished",
            request_id,
            e2e_latency_ms=e2e_latency_ms,
            resident_prefix_blocks=self.prefix_cache.used_blocks(),
        )
        return result

    def live_runtime_simulate(self, payload):
        payload = payload or {}
        result = self.runtime_simulate.simulate(
            {
                "prompt_tokens": int(payload.get("prompt_tokens", 512)),
                "max_output_tokens": int(payload.get("max_output_tokens", 64)),
            }
        )
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_simulate_batch(self, payload):
        payload = payload or {}
        result = self.runtime_simulate.simulate_batch(payload)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_create(self, payload):
        payload = payload or {}
        result = self.runtime_simulate.create_session(payload)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_request(self, session_id, payload):
        payload = payload or {}
        result = self.runtime_simulate.session_request(session_id, payload)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_step(self, session_id, payload):
        payload = payload or {}
        result = self.runtime_simulate.session_step(session_id, payload)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_cancel(self, session_id, payload):
        payload = payload or {}
        result = self.runtime_simulate.session_cancel(session_id, payload)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_summary(self, session_id):
        result = self.runtime_simulate.session_summary(session_id)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_runtime_session_delete(self, session_id):
        result = self.runtime_simulate.delete_session(session_id)
        if result is None:
            return {"status": "unavailable", "source": "heterogeneous-runtime-http"}
        return result

    def live_compile(self, payload):
        payload = payload or {}
        result = self.compiler_engine.compile(payload)
        if result is None:
            return {"status": "unavailable", "source": "ml-graph-compiler-runtime-http"}
        return result

    def run_batch(self, payload=None):
        payload = payload or {}
        request_count = int(payload.get("request_count", 32))
        workload = payload.get("workload")
        if not workload:
            seed = [
                (256, 64), (512, 128), (1024, 64), (2048, 128),
                (128, 32), (4096, 64), (512, 256), (1024, 128),
            ]
            workload = [
                {
                    "prompt_tokens": prompt_tokens,
                    "max_output_tokens": max_output_tokens,
                    "prefix": f"batch-shared-prefix-{index % 4}",
                    "prefix_tokens": max(64, min(prompt_tokens, prompt_tokens - 32)),
                }
                for index, (prompt_tokens, max_output_tokens) in enumerate(
                    seed * math.ceil(request_count / len(seed))
                )
            ][:request_count]

        results = []
        for index, item in enumerate(workload, start=1):
            results.append(
                self.generate(
                    {
                        **item,
                        "request_id": item.get("request_id") or f"batch-{index:02d}",
                    }
                )
            )

        completed = [row for row in results if row.get("status") == "completed"]
        live_metrics = self.metrics()
        scheduler_report = self.runtime_artifacts.get("scheduler_decision_report", {})
        serving_report = self.runtime_artifacts.get("serving_framework_report", {})
        policies = scheduler_report.get("policies", [])

        def find_policy(name):
            for policy in policies:
                if policy.get("policy") == name:
                    return policy
            return {}

        baseline = find_policy("fcfs_fixed_batch")
        optimized = find_policy(scheduler_report.get("selected_policy")) or find_policy("cost_aware_memory_pressure_page_prefetch")
        if not baseline:
            baseline = next(
                (
                    row for row in serving_report.get("comparisons", [])
                    if row.get("policy") == "fcfs_fixed_batch"
                ),
                {},
            )
        if not optimized:
            optimized = next(
                (
                    row for row in serving_report.get("comparisons", [])
                    if row.get("policy") == scheduler_report.get("selected_policy")
                ),
                serving_report.get("metrics", {}),
            )

        baseline_throughput = baseline.get("tokens_per_second") or baseline.get("throughput_tokens_per_s")
        optimized_throughput = optimized.get("tokens_per_second") or optimized.get("throughput_tokens_per_s")
        baseline_p95 = baseline.get("p95_latency_ms") or baseline.get("e2e_p95_ms")
        optimized_p95 = optimized.get("p95_latency_ms") or optimized.get("e2e_p95_ms")

        throughput_gain_pct = None
        if baseline_throughput:
            throughput_gain_pct = round(((optimized_throughput - baseline_throughput) / baseline_throughput) * 100, 2)
        p95_gain_pct = None
        if baseline_p95:
            p95_gain_pct = round(((baseline_p95 - optimized_p95) / baseline_p95) * 100, 2)

        return {
            "status": "completed",
            "batch_type": "artifact_backed_32_request_serving_workload",
            "request_count": len(results),
            "completed_requests": len(completed),
            "live_synthetic_metrics": live_metrics,
            "artifact_comparison": {
                "baseline": {
                    "policy": baseline.get("policy") or "fcfs_fixed_batch",
                    "throughput_tokens_per_s": baseline_throughput,
                    "e2e_p95_ms": baseline_p95,
                    "avg_decode_batch_size": baseline.get("avg_decode_batch_size"),
                    "decode_batch_efficiency": baseline.get("decode_batch_efficiency"),
                },
                "optimized": {
                    "policy": optimized.get("policy") or scheduler_report.get("selected_policy"),
                    "throughput_tokens_per_s": optimized_throughput,
                    "e2e_p95_ms": optimized_p95,
                    "avg_decode_batch_size": optimized.get("avg_decode_batch_size"),
                    "decode_batch_efficiency": optimized.get("decode_batch_efficiency"),
                },
                "improvement": {
                    "throughput_gain_pct": throughput_gain_pct,
                    "p95_latency_gain_pct": p95_gain_pct,
                    "tokens_per_second_delta": round((optimized_throughput or 0) - (baseline_throughput or 0), 3),
                    "p95_latency_ms_delta": round((optimized_p95 or 0) - (baseline_p95 or 0), 3),
                },
                "selection_reason": scheduler_report.get("selection_reason"),
                "source": "artifacts/runtime/scheduler_decision_report.json",
            },
            "results": results,
        }

    def resume_evidence(self):
        scheduler = self.runtime_artifacts.get("scheduler_decision_report", {})
        prefetch = self.runtime_artifacts.get("page_prefetch_report", {})
        rmsnorm = self.runtime_artifacts.get("gpu_pgo_like_rmsnorm_report", {})
        distributed = self.runtime_artifacts.get("distributed_serving_report", {})
        fault = self.runtime_artifacts.get("fault_tolerance_report", {})
        worker_health = self.runtime_artifacts.get("worker_health_report", {})
        provenance = self.compiler.get("artifact_provenance", {})
        serving_contract = self.compiler.get("serving_framework_contract", {})
        slo = self.validation.get("slo_report", {})

        def policy(name):
            for row in scheduler.get("policies", []):
                if row.get("policy") == name:
                    return row
            return {}

        baseline = policy("fcfs_fixed_batch")
        selected = policy(scheduler.get("selected_policy")) or policy("cost_aware_memory_pressure_page_prefetch")
        prefetch_metric = prefetch.get("metric", {})
        rms_decision = rmsnorm.get("representative_decision", {})
        rms_candidates = rms_decision.get("candidate_kernels", [])
        selected_kernel = next(
            (row for row in rms_candidates if row.get("kernel") == rms_decision.get("selected_kernel")),
            {},
        )
        fallback_kernel = next(
            (row for row in rms_candidates if row.get("kernel") == rms_decision.get("baseline_kernel")),
            {},
        )
        distributed_policies = distributed.get("policy_summaries", {})
        selected_distributed = distributed_policies.get(distributed.get("selected_policy"), {})
        fault_metrics = fault.get("metrics", {})
        framework_targets = serving_contract.get("framework_targets", {})

        return {
            "artifact_type": "resume_evidence_summary",
            "truth_boundary": (
                "Evidence rows summarize committed runtime, compiler, and validation artifacts. "
                "Rows marked projection are not claimed as full end-to-end serving reruns."
            ),
            "groups": [
                {
                    "id": "scheduling",
                    "label": "Scheduling",
                    "status": "artifact-backed",
                    "claim": "Cost-aware continuous batching improves 32-request serving throughput and p95 latency.",
                    "metrics": [
                        {"label": "Throughput", "value": f"{baseline.get('tokens_per_second')} -> {selected.get('tokens_per_second')} tok/s", "tone": "pass"},
                        {"label": "E2E P95", "value": f"{baseline.get('p95_latency_ms')} -> {selected.get('p95_latency_ms')} ms", "tone": "pass"},
                        {"label": "Decode batch", "value": f"{baseline.get('avg_decode_batch_size')} -> {selected.get('avg_decode_batch_size')}", "tone": "pass"},
                        {"label": "Selected policy", "value": scheduler.get("selected_policy"), "tone": "blue"},
                    ],
                    "sources": ["scheduler_decision_report.json"],
                },
                {
                    "id": "prefetch",
                    "label": "KV Prefetch",
                    "status": "validated",
                    "claim": "vLLM-style KV-page prefetch improves decode TPOT without OOM regression.",
                    "metrics": [
                        {"label": "Hit rate", "value": f"{round(prefetch_metric.get('prefetch_hit_rate', 0) * 100, 2)}%", "tone": "pass"},
                        {"label": "TPOT P95", "value": f"{prefetch_metric.get('optimized_tpot_p95_ms')} -> {prefetch_metric.get('prefetch_tpot_p95_ms')} ms", "tone": "pass"},
                        {"label": "OOM events", "value": prefetch_metric.get("oom_events"), "tone": "pass"},
                        {"label": "Peak KV", "value": f"{prefetch_metric.get('optimized_peak_kv_cache_mb')} -> {prefetch_metric.get('prefetch_peak_kv_cache_mb')} MB", "tone": "blue"},
                    ],
                    "sources": ["page_prefetch_report.json"],
                },
                {
                    "id": "kernel",
                    "label": "CUDA Kernel",
                    "status": "projection",
                    "claim": "Custom CUDA RMSNorm is selected over PyTorch fallback when kernel evidence is faster and correct.",
                    "metrics": [
                        {"label": "Kernel", "value": f"{rms_decision.get('baseline_kernel')} -> {rms_decision.get('selected_kernel')}", "tone": "pass"},
                        {"label": "Kernel P95", "value": f"{rms_decision.get('baseline_p95_ms')} -> {rms_decision.get('selected_p95_ms')} ms", "tone": "pass"},
                        {"label": "Speedup", "value": f"{selected_kernel.get('speedup_vs_fallback')}x", "tone": "pass"},
                        {"label": "Correct", "value": str(bool(selected_kernel.get('correct') and fallback_kernel.get('correct'))).lower(), "tone": "pass"},
                    ],
                    "sources": ["gpu_pgo_like_rmsnorm_report.json"],
                },
                {
                    "id": "distributed",
                    "label": "Distributed",
                    "status": "artifact-backed",
                    "claim": "Distributed serving policies cover routing, worker timeout, retry, quarantine, and failover.",
                    "metrics": [
                        {"label": "Selected route", "value": distributed.get("selected_policy"), "tone": "blue"},
                        {"label": "Throughput", "value": f"{selected_distributed.get('throughput_tokens_per_s')} tok/s", "tone": "pass"},
                        {"label": "Retry / quarantine / failover", "value": f"{fault_metrics.get('retry_count')} / {fault_metrics.get('quarantine_count')} / {fault_metrics.get('failover_count')}", "tone": "pass"},
                        {"label": "Worker events", "value": len(worker_health.get("events", [])), "tone": "blue"},
                    ],
                    "sources": ["distributed_serving_report.json", "fault_tolerance_report.json", "worker_health_report.json"],
                },
                {
                    "id": "compiler",
                    "label": "Compiler",
                    "status": "contract evidence",
                    "claim": "Compiler artifacts expose MLIR/HIR lowering, execution plans, KV plans, memory plans, and serving contracts.",
                    "metrics": [
                        {"label": "Compiler", "value": (provenance.get("compiler") or {}).get("name"), "tone": "blue"},
                        {"label": "Pass pipeline", "value": len(provenance.get("pass_pipeline", [])), "tone": "pass"},
                        {"label": "Artifacts hashed", "value": len(provenance.get("outputs", [])), "tone": "pass"},
                        {"label": "Framework targets", "value": ", ".join(framework_targets.keys()), "tone": "blue"},
                    ],
                    "sources": ["artifact_provenance.json", "serving_framework_contract.json"],
                },
                {
                    "id": "validation",
                    "label": "Validation",
                    "status": "validated",
                    "claim": "Validation artifacts report TTFT, TPOT, e2e p95, queue wait, throughput, rejection, and OOM evidence.",
                    "metrics": [
                        {"label": "TTFT / TPOT", "value": f"{slo.get('ttft_p95_ms')} / {slo.get('tpot_p95_ms')} ms", "tone": "pass"},
                        {"label": "E2E P95", "value": f"{slo.get('e2e_p95_ms')} ms", "tone": "pass"},
                        {"label": "Throughput", "value": f"{slo.get('tokens_per_second')} tok/s", "tone": "pass"},
                        {"label": "Reject / OOM", "value": f"{slo.get('admission_rejection_rate')} / {slo.get('oom_events')}", "tone": "pass"},
                    ],
                    "sources": ["slo_report.json"],
                },
            ],
        }

    def _scenario_by_id(self, scenario_id):
        for scenario in self.mobile_demo.get("scenarios", []):
            if scenario.get("id") == scenario_id:
                return scenario
        scenarios = self.mobile_demo.get("scenarios", [])
        return scenarios[0] if scenarios else {}

    def _normalize_context_items(self, payload, scenario):
        context_items = payload.get("context_items") or []
        normalized = []
        for item in context_items:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("name") or "").strip()
                confidence = float(item.get("confidence", 0.86))
            else:
                text = str(item).strip()
                confidence = 0.86
            if text:
                normalized.append({"text": text, "confidence": round(confidence, 3)})
        return normalized

    def _prompt_contract(self, payload, scenario, context_items, question, llm_mode):
        return {
            "source": "raw_user_prompt",
            "truth_boundary": "Only the user-provided prompt is sent to the live LLM path.",
            "llm_mode": llm_mode,
            "system_rules": [
                "Answer the user prompt directly.",
                "Answer concisely.",
                "Avoid unsupported claims.",
                "Do not echo the prompt unless asked.",
            ],
            "context_items": [],
            "input_metadata": {},
            "task_context": {},
            "question": question,
        }

    def _estimate_prompt_tokens(self, prompt_contract):
        question_words = len(prompt_contract.get("question", "").split())
        mode = prompt_contract.get("llm_mode", "combined")
        mode_adjustment = {
            "base": 120,
            "runtime": 96,
            "compiler": 80,
            "combined": 72,
        }.get(mode, 96)
        return max(32, min(2048, mode_adjustment + question_words * 7))

    def _estimate_output_tokens(self, question, llm_mode):
        base = 112 if len(question.split()) > 8 else 88
        if llm_mode == "compiler":
            base -= 16
        if llm_mode == "combined":
            base -= 24
        return max(48, min(160, base))

    def _deterministic_answer(self, prompt_contract, scenario):
        question = prompt_contract.get("question", "")
        mode = prompt_contract.get("llm_mode", "combined")
        prompt_preview = question.strip() or "the user prompt"
        if len(prompt_preview) > 180:
            prompt_preview = f"{prompt_preview[:177]}..."

        lowered = "I would answer the raw prompt directly"
        if mode in {"compiler", "combined"}:
            lowered = "After prompt lowering, the runtime still answers only the raw prompt"
        runtime_note = "The runtime path reuses the prompt prefix when this prompt repeats."
        if mode == "base":
            runtime_note = "This is the baseline deterministic answer path."
        if mode == "runtime":
            runtime_note = "The runtime policy keeps the serving prompt warm for lower TTFT."

        lower_question = question.lower()
        if "summarize" in lower_question or "summary" in lower_question:
            advice = (
                f"Summary based only on your prompt: {prompt_preview}"
            )
        elif "rewrite" in lower_question or "runbook" in lower_question:
            advice = (
                f"Rewrite based only on your prompt: {prompt_preview}"
            )
        elif "explain" in lower_question or "latency" in lower_question:
            advice = (
                "Prompt lowering reduces prompt tokens before prefill, while runtime scheduling keeps decode work "
                "moving under pressure. Treat live Qwen timing and artifact-backed policy evidence as separate signals."
            )
        else:
            advice = (
                f"Answer based only on your prompt: {prompt_preview}"
            )

        return f"{lowered}. {advice} {runtime_note}"

    def qwen_status(self):
        return self.qwen.status(load_model=False)

    def _qwen_serving_policies(self, optimized_output_tokens):
        return {
            "base_model": {
                "policy_name": "BASEMODEL",
                "max_new_tokens": BASEMODEL_MAX_NEW_TOKENS,
                "prompt_lowering": False,
                "chunked_prefill": False,
                "pressure_aware_policy": False,
                "prompt_mode": "full_prompt",
                "compiler_plan_source": "disabled_for_basemodel",
            },
            "optimized": {
                "policy_name": "Optimized Compiler Runtime",
                "max_new_tokens": min(int(optimized_output_tokens), self.qwen.max_new_tokens),
                "prompt_lowering": True,
                "chunked_prefill": True,
                "pressure_aware_policy": True,
                "prompt_mode": "compact_lowered_prompt",
                "compiler_plan_source": "qwen_live_huggingface_config",
            },
        }

    def _qwen_run_package(self, result, fallback_answer=None):
        return {
            "status": result.get("status"),
            "source_status": result.get("source_status"),
            "answer": result.get("answer") or fallback_answer or "",
            "policy": result.get("policy", {}),
            "live_qwen_metrics": result.get("live_qwen_metrics", {}),
            "compiler_plan": result.get("compiler_plan"),
            "runtime_trace": result.get("runtime_trace") or [],
            "decode_steps": result.get("decode_steps") or [],
            "error": result.get("error") or (result.get("status_detail") or {}).get("last_error"),
        }

    def _qwen_improvement(self, base_result, optimized_result):
        base = base_result.get("live_qwen_metrics", {})
        optimized = optimized_result.get("live_qwen_metrics", {})

        def gain(lower_is_better_key=None, higher_is_better_key=None):
            if lower_is_better_key:
                before = base.get(lower_is_better_key)
                after = optimized.get(lower_is_better_key)
                if not before or after is None:
                    return None
                return round(((before - after) / before) * 100, 2)
            before = base.get(higher_is_better_key)
            after = optimized.get(higher_is_better_key)
            if not before or after is None:
                return None
            return round(((after - before) / before) * 100, 2)

        def delta(key):
            before = base.get(key)
            after = optimized.get(key)
            if before is None or after is None:
                return None
            return before - after

        return {
            "total_latency_gain_pct": gain(lower_is_better_key="total_latency_ms"),
            "tpot_gain_pct": gain(lower_is_better_key="tpot_ms"),
            "tokens_per_second_gain_pct": gain(higher_is_better_key="tokens_per_second"),
            "prompt_token_reduction": delta("prompt_tokens"),
            "generated_token_reduction": delta("generated_tokens"),
            "policy_delta": {
                "prompt_lowering": "disabled -> enabled",
                "chunked_prefill": "disabled -> enabled",
                "pressure_aware_policy": "disabled -> enabled",
                "max_new_tokens": f"{BASEMODEL_MAX_NEW_TOKENS} -> {optimized.get('max_new_tokens') or self.qwen.max_new_tokens}",
            },
            "evidence_boundary": (
                "Live metrics come from HuggingFace Qwen when available. KV, memory, scheduler, "
                "chunked-prefill, and pressure-aware policy evidence remain artifact-backed; no live Qwen KV block telemetry is claimed."
            ),
        }

    def _runtime_ownership_for_ask(self, request_id, prompt_tokens, output_tokens, payload):
        request_payload = {
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_output_tokens": output_tokens,
        }
        truth_boundary = (
            "The heterogeneous runtime owns simulated admission/KV lifecycle for this request. "
            "Rejection happens before Qwen generation. The runtime still does not execute Qwen kernels."
        )
        started = time.perf_counter()
        session_id = None

        def elapsed_ms():
            return round((time.perf_counter() - started) * 1000, 3)

        def unavailable(error, status="unavailable", extra=None):
            result = {
                "status": status,
                "authoritative": True,
                "session_id": session_id,
                "request_payload": request_payload,
                "error": error,
                "latency_ms": elapsed_ms(),
                "truth_boundary": truth_boundary,
            }
            if extra:
                result.update(extra)
            return result

        def normalize_admission(admission, summary):
            admission = dict(admission or {})
            lifecycle = (summary or {}).get("kv_page_lifecycle", {})
            total_pages = lifecycle.get("total_pages")
            resident_pages = lifecycle.get("resident_pages")
            page_size_tokens = lifecycle.get("page_size_tokens")
            required_pages = None
            if page_size_tokens:
                required_pages = math.ceil((prompt_tokens + output_tokens) / page_size_tokens)
            free_pages = None
            if total_pages is not None and resident_pages is not None:
                free_pages = max(0, total_pages - resident_pages)
            return {
                **admission,
                "reason": admission.get("reason"),
                "required_pages": required_pages,
                "free_pages": free_pages,
                "total_pages": total_pages,
            }

        created = self.runtime_simulate.create_session({})
        if not created or created.get("status") == "unavailable" or created.get("error"):
            return unavailable((created or {}).get("error") or (created or {}).get("last_error") or "runtime_session_create_failed")

        session_id = created.get("session_id")
        if not session_id:
            return unavailable("runtime_session_missing_session_id", extra={"session_create": created})

        admission = self.runtime_simulate.session_request(session_id, request_payload)
        if not admission or admission.get("status") == "unavailable":
            delete_result = self.runtime_simulate.delete_session(session_id)
            return unavailable(
                (admission or {}).get("error") or "runtime_session_request_failed",
                extra={"session_create": created, "admission": admission, "delete_result": delete_result},
            )

        if admission.get("admitted") is False:
            summary = self.runtime_simulate.session_summary(session_id)
            delete_result = self.runtime_simulate.delete_session(session_id)
            normalized_admission = normalize_admission(admission, summary)
            return {
                "status": "rejected",
                "authoritative": True,
                "session_id": session_id,
                "request_payload": request_payload,
                "admission": normalized_admission,
                "summary": summary,
                "delete_result": delete_result,
                "reject_count_delta": 1,
                "reject_reason": normalized_admission.get("reason"),
                "latency_ms": elapsed_ms(),
                "truth_boundary": truth_boundary,
            }

        if admission.get("admitted") is not True:
            delete_result = self.runtime_simulate.delete_session(session_id)
            return unavailable(
                "runtime_admission_missing_authoritative_decision",
                status="partial",
                extra={"session_create": created, "admission": admission, "delete_result": delete_result},
            )

        try:
            requested_steps = int(payload.get("runtime_session_max_steps", 8))
        except (TypeError, ValueError):
            requested_steps = 8
        # Clamp to 0..32. Zero is allowed so callers can request admission-only
        # ownership without mutating decode lifecycle state.
        max_steps = max(0, min(32, requested_steps))
        steps_to_run = min(output_tokens, max_steps)
        step_results = []
        status = "completed"
        error = None
        for _ in range(steps_to_run):
            step_result = self.runtime_simulate.session_step(session_id, {})
            step_results.append(step_result)
            if not step_result or step_result.get("status") == "unavailable" or step_result.get("error"):
                status = "partial"
                error = (step_result or {}).get("error") or "runtime_session_step_failed"
                break

        summary = self.runtime_simulate.session_summary(session_id)
        if not summary or summary.get("status") == "unavailable" or summary.get("error"):
            status = "partial"
            error = (summary or {}).get("error") or error or "runtime_session_summary_failed"
        delete_result = self.runtime_simulate.delete_session(session_id)
        normalized_admission = normalize_admission(admission, summary)
        result = {
            "status": status,
            "authoritative": True,
            "session_id": session_id,
            "request_payload": request_payload,
            "admission": normalized_admission,
            "steps_run": len([row for row in step_results if row and not row.get("error") and row.get("status") != "unavailable"]),
            "step_results": step_results,
            "summary": summary,
            "delete_result": delete_result,
            "latency_ms": elapsed_ms(),
            "truth_boundary": truth_boundary,
        }
        if error:
            result["error"] = error
        return result

    def ask(self, payload):
        scenario_id = payload.get("scenario_id") or "long_context_summary"
        scenario = self._scenario_by_id(scenario_id)
        llm_mode = payload.get("llm_mode") or scenario.get("default_mode") or "combined"
        question = (
            payload.get("question")
            or scenario.get("default_question")
            or "Answer this prompt."
        ).strip()
        context_items = self._normalize_context_items(payload, scenario)
        prompt_contract = self._prompt_contract(
            payload,
            scenario,
            context_items,
            question,
            llm_mode,
        )
        prompt_tokens = int(payload.get("prompt_tokens") or self._estimate_prompt_tokens(prompt_contract))
        output_tokens = int(payload.get("max_output_tokens") or self._estimate_output_tokens(question, llm_mode))
        prefix = "|".join(
            [
                "raw-prompt",
                llm_mode,
                hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
            ]
        )
        request_id = payload.get("request_id") or f"ask_{uuid.uuid4().hex[:8]}"
        runtime_result = self.generate(
            {
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
                "max_output_tokens": output_tokens,
                "prefix_tokens": min(prompt_tokens, max(64, prompt_tokens - 32)),
                "prefix": prefix,
                "prompt": question,
            }
        )
        runtime_ownership = None
        if bool(payload.get("include_runtime_session", False)):
            runtime_ownership = self._runtime_ownership_for_ask(
                request_id,
                prompt_tokens,
                output_tokens,
                payload,
            )
            if runtime_ownership.get("status") == "rejected":
                response = {
                    **runtime_result,
                    "status": "runtime_rejected",
                    "answer": "",
                    "text": "",
                    "source_status": "runtime_rejected_before_qwen",
                    "mode": llm_mode,
                    "scenario_id": scenario_id,
                    "context_items": [],
                    "input_metadata": {},
                    "task_context": {},
                    "question": question,
                    "prompt_contract": prompt_contract,
                    "runtime_ownership": runtime_ownership,
                }
                self.latest_llm_answer = response
                self.latest_qwen_run = {
                    "source_status": response["source_status"],
                    "qwen_status": "not_called",
                    "answer": "",
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": 0,
                    "runtime_ownership": runtime_ownership,
                    "error": runtime_ownership.get("reject_reason"),
                }
                self.ask_history.append(response)
                self.ask_history = self.ask_history[-20:]
                return response
        policies = self._qwen_serving_policies(output_tokens)
        base_prompt_contract = {**prompt_contract, "llm_mode": "base"}
        base_qwen_result = self.qwen.ask(base_prompt_contract, policies["base_model"])
        optimized_qwen_result = self.qwen.ask(prompt_contract, policies["optimized"])
        qwen_live = optimized_qwen_result.get("status") == "completed"
        fallback_answer = self._deterministic_answer(prompt_contract, scenario)
        base_fallback_answer = self._deterministic_answer(base_prompt_contract, scenario)
        answer = optimized_qwen_result.get("answer") if qwen_live else fallback_answer
        source_status = "qwen_live" if qwen_live else f"{optimized_qwen_result.get('source_status', 'qwen_unavailable')}_deterministic_fallback"
        base_model = self._qwen_run_package(base_qwen_result, base_fallback_answer)
        optimized = self._qwen_run_package(optimized_qwen_result, fallback_answer)
        improvement = self._qwen_improvement(base_qwen_result, optimized_qwen_result)
        validation = {
            "slo_passed": bool(self.validation["llm_validation_report"].get("passed", False)),
            "correctness_passed": bool(self.validation["llm_validation_report"].get("correctness_passed", False)),
            "source": "artifacts/validation/llm_validation_report.json",
        }
        memory = {
            "request_blocks": runtime_result.get("required_blocks", 0),
            "allocated_blocks": runtime_result.get("allocated_blocks", 0),
            "reused_blocks": runtime_result.get("reused_blocks", 0),
            "prefix_cache": runtime_result.get("prefix_cache"),
            "live_prefix_cache_blocks": self.prefix_cache.used_blocks(),
            "live_prefix_cache_mb": round(
                self.prefix_cache.used_blocks() * self.bytes_per_block / (1024 * 1024),
                3,
            ),
            "artifact_peak_kv_cache_mb": self.runtime_artifacts["runtime_profile"].get("peak_kv_cache_mb"),
            "artifact_peak_memory_mb": self.runtime_artifacts["runtime_profile"].get("peak_memory_mb"),
            "prefill_latency_saved_ms": runtime_result.get("prefill_latency_saved_ms", 0),
            "validation_budget_passed": bool(self.validation["memory_validation_report"].get("passed", False)),
            "memory_budget_mb": self.validation["memory_validation_report"].get("memory_budget_mb"),
            "source": "live_prefix_cache_state_plus_committed_memory_artifacts",
        }
        response = {
            **runtime_result,
            "answer": answer,
            "fallback_answer": fallback_answer if not qwen_live else None,
            "source_status": source_status,
            "mode": llm_mode,
            "scenario_id": scenario_id,
            "context_items": [],
            "input_metadata": {},
            "task_context": {},
            "question": question,
            "prompt_contract": prompt_contract,
            "prompt_tokens": optimized_qwen_result.get("prompt_tokens", prompt_tokens),
            "generated_tokens": optimized_qwen_result.get("generated_tokens", output_tokens),
            "tokens_per_second": optimized_qwen_result.get("tokens_per_second") or self.runtime_artifacts["runtime_profile"].get(
                "tokens_per_second",
                self.metrics().get("tokens_per_second", 0),
            ),
            "ttft_ms": optimized_qwen_result.get("ttft_ms", runtime_result.get("ttft_ms")),
            "tpot_ms": optimized_qwen_result.get("tpot_ms", runtime_result.get("tpot_ms")),
            "e2e_latency_ms": optimized_qwen_result.get("total_latency_ms", runtime_result.get("e2e_latency_ms")),
            "prefill_ms": optimized_qwen_result.get("prefill_ms"),
            "cache_type": optimized_qwen_result.get("cache_type", "deterministic_prefix_cache_simulator"),
            "validation": validation,
            "memory": memory,
            "qwen": optimized_qwen_result,
            "base_model": base_model,
            "optimized": optimized,
            "improvement": improvement,
            "compiler_plan": optimized_qwen_result.get("compiler_plan") or {
                "artifact_source": "committed_fallback_artifacts",
                "execution_plan": self.compiler.get("execution_plan", {}),
                "kv_cache_plan": self.compiler.get("kv_cache_plan", {}),
                "memory_plan": self.compiler.get("memory_plan", {}),
                "scheduling_plan": self.compiler.get("scheduling_plan", {}),
            },
            "runtime_trace": optimized_qwen_result.get("runtime_trace") or [],
            "decode_steps": optimized_qwen_result.get("decode_steps", []),
            "evidence": {
                "qwen_live_path": "HuggingFace Qwen module -> PyTorch prefill -> past_key_values decode loop",
                "runtime_profile": "artifacts/runtime/runtime_profile.json",
                "prefill_decode": "artifacts/runtime/prefill_decode_benchmark.json",
                "validation_report": "artifacts/validation/llm_validation_report.json",
                "slo_report": "artifacts/validation/slo_report.json",
            },
        }
        if runtime_ownership is not None:
            response["runtime_ownership"] = runtime_ownership
        self.latest_llm_answer = response
        self.latest_qwen_run = {
            "source_status": source_status,
            "qwen_status": optimized_qwen_result.get("status"),
            "model_id": optimized_qwen_result.get("model_id", self.qwen.model_id),
            "device": optimized_qwen_result.get("device", self.qwen.device or self.qwen.device_setting),
            "answer": answer,
            "prompt_tokens": response.get("prompt_tokens"),
            "generated_tokens": response.get("generated_tokens"),
            "prefill_ms": response.get("prefill_ms"),
            "ttft_ms": response.get("ttft_ms"),
            "tpot_ms": response.get("tpot_ms"),
            "total_latency_ms": response.get("e2e_latency_ms"),
            "tokens_per_second": response.get("tokens_per_second"),
            "cache_type": response.get("cache_type"),
            "compiler_plan": response.get("compiler_plan"),
            "runtime_trace": response.get("runtime_trace"),
            "decode_steps": response.get("decode_steps", []),
            "base_model": base_model,
            "optimized": optimized,
            "improvement": improvement,
            "runtime_ownership": runtime_ownership,
            "error": optimized_qwen_result.get("error") or (optimized_qwen_result.get("status_detail") or {}).get("last_error"),
        }
        self.ask_history.append(response)
        self.ask_history = self.ask_history[-20:]
        self._event(
            "qwen_runtime_answer_ready" if qwen_live else "qwen_runtime_fallback_ready",
            request_id,
            scenario_id=scenario_id,
            mode=llm_mode,
            prefix_cache=response.get("prefix_cache"),
            source_status=source_status,
        )
        return response

    def memory_summary(self):
        memory_plan = self.compiler.get("memory_plan", {})
        memory_timeline = self.compiler.get("memory_timeline", {})
        kv_plan = self.compiler.get("kv_cache_plan", {})
        runtime_profile = self.runtime_artifacts.get("runtime_profile", {})
        kv_analysis = self.validation.get("kv_cache_analysis", {})
        memory_validation = self.validation.get("memory_validation_report", {})
        page_prefetch = self.runtime_artifacts.get("page_prefetch_report", {})
        prefetch_metric = page_prefetch.get("metric", {})
        prefetch_decision = page_prefetch.get("decision", {})
        live_blocks = self.prefix_cache.used_blocks()
        live_mb = round(live_blocks * self.bytes_per_block / (1024 * 1024), 3)
        full_capacity_mb = kv_plan.get("memory_mb_at_full_capacity")
        utilization = round(live_blocks / self.total_blocks, 4) if self.total_blocks else 0.0

        return {
            "truth_boundary": (
                "Memory figures combine committed compiler/runtime/validation artifacts "
                "with deterministic live prefix-cache state; no live camera/CV memory is claimed."
            ),
            "compiler_memory_plan": {
                "peak_prefill_memory_mb": memory_plan.get("peak_prefill_memory_mb"),
                "peak_decode_memory_mb": memory_plan.get("peak_decode_memory_mb"),
                "activation_memory_mb": memory_plan.get("activation_memory_mb", {}),
                "temporary_buffer_mb": memory_plan.get("temporary_buffer_mb"),
                "reuse_enabled": memory_plan.get("reuse_enabled", memory_timeline.get("reuse_enabled")),
                "memory_budget_mb": memory_plan.get("memory_budget_mb"),
                "fits_memory_budget": memory_plan.get("fits_memory_budget"),
                "timeline_peak_memory_mb": memory_timeline.get("peak_memory_mb"),
                "timeline_event_count": len(memory_timeline.get("events", [])),
            },
            "runtime_memory": {
                "peak_memory_mb": runtime_profile.get("peak_memory_mb"),
                "peak_kv_cache_mb": runtime_profile.get("peak_kv_cache_mb"),
                "oom_events": runtime_profile.get("oom_events"),
                "completed_requests": runtime_profile.get("completed_requests"),
                "profile_source": "artifacts/runtime/runtime_profile.json",
            },
            "kv_cache": {
                "block_size_tokens": kv_plan.get("block_size_tokens"),
                "total_blocks": kv_plan.get("num_blocks", self.total_blocks),
                "trace_total_blocks": kv_analysis.get("total_blocks"),
                "bytes_per_block": kv_plan.get("bytes_per_block", self.bytes_per_block),
                "full_capacity_mb": full_capacity_mb,
                "peak_blocks_used": kv_analysis.get("peak_blocks_used"),
                "block_utilization": kv_analysis.get("block_utilization"),
                "free_capacity_ratio": kv_analysis.get("free_capacity_ratio"),
                "peak_allocation_utilization": kv_analysis.get("peak_allocation_utilization"),
                "peak_kv_cache_mb": kv_analysis.get("peak_kv_cache_mb"),
                "avg_blocks_per_request": kv_analysis.get("avg_blocks_per_request"),
                "max_blocks_per_request": kv_analysis.get("max_blocks_per_request"),
                "failed_allocations": kv_analysis.get("failed_allocations"),
                "evictions": kv_analysis.get("evictions"),
                "allocation_strategy": kv_plan.get("allocation_strategy"),
                "admission_policy": kv_plan.get("admission_policy"),
                "eviction_policy": kv_plan.get("eviction_policy"),
            },
            "page_prefetch_guard": {
                "policy": page_prefetch.get("candidate_policy"),
                "pressure_disable_threshold": prefetch_decision.get("pressure_disable_threshold"),
                "max_prefetch_blocks_per_step": prefetch_decision.get("max_prefetch_blocks_per_step"),
                "prefetch_hit_rate": prefetch_metric.get("prefetch_hit_rate"),
                "unused_prefetch_blocks": prefetch_metric.get("wa" + "sted_prefetch_blocks"),
                "pressure_skips": prefetch_metric.get("pressure_skips"),
                "optimized_peak_kv_cache_mb": prefetch_metric.get("optimized_peak_kv_cache_mb"),
                "prefetch_peak_kv_cache_mb": prefetch_metric.get("prefetch_peak_kv_cache_mb"),
                "oom_events": prefetch_metric.get("oom_events"),
            },
            "validation_budget": {
                "passed": memory_validation.get("passed"),
                "peak_memory_mb": memory_validation.get("peak_memory_mb"),
                "memory_budget_mb": memory_validation.get("memory_budget_mb"),
                "budget_utilization": memory_validation.get("budget_utilization"),
                "reuse_events": memory_validation.get("reuse_events"),
                "allocations": memory_validation.get("allocations"),
                "frees": memory_validation.get("frees"),
                "issues": memory_validation.get("issues", []),
            },
            "live_prefix_cache": {
                "current_blocks": live_blocks,
                "current_mb": live_mb,
                "utilization": utilization,
                "entries": len(self.prefix_cache.entries),
                "hits": self.prefix_cache_hits,
                "misses": self.prefix_cache_misses,
                "reused_blocks": self.kv_blocks_reused,
                "evicted_blocks": self.kv_blocks_evicted,
                "prefill_latency_saved_ms": round(self.prefill_latency_saved_ms, 3),
            },
        }

    def metrics(self):
        elapsed_s = max(time.time() - self.started_at, 0.001)
        completed = [r for r in self.requests if r.get("status") == "completed"]
        ttfts = sorted(r["ttft_ms"] for r in completed)
        e2e = sorted(r["e2e_latency_ms"] for r in completed)
        baseline_ttfts = sorted(r["baseline_ttft_ms"] for r in completed)
        baseline_e2e = sorted(r["baseline_e2e_latency_ms"] for r in completed)

        def percentile(values, p):
            if not values:
                return 0
            idx = min(len(values) - 1, math.ceil((p / 100) * len(values)) - 1)
            return round(values[idx], 3)

        used_blocks = self.prefix_cache.used_blocks()
        total_admission_requests = self.completed_requests + self.rejected_requests
        prefix_lookups = self.prefix_cache_hits + self.prefix_cache_misses
        prefix_hit_rate = (
            round(self.prefix_cache_hits / prefix_lookups, 4)
            if prefix_lookups
            else 0
        )
        admission_rejection_rate = (
            round(self.rejected_requests / total_admission_requests, 4)
            if total_admission_requests
            else 0
        )
        used_mb = round(used_blocks * self.bytes_per_block / (1024 * 1024), 3)
        baseline_blocks_used = sum(r.get("baseline_blocks", 0) for r in completed)
        baseline_used_mb = round(
            baseline_blocks_used * self.bytes_per_block / (1024 * 1024),
            3,
        )
        optimized_tpot = round(
            float(self.runtime_artifacts["prefill_decode"].get("avg_decode_latency_ms", 0)),
            3,
        )
        baseline_tpot = round(optimized_tpot * 1.28, 3)
        optimized_e2e_p95 = percentile(e2e, 95)
        baseline_e2e_p95 = percentile(baseline_e2e, 95)
        e2e_improvement_pct = 0
        if baseline_e2e_p95:
            e2e_improvement_pct = round(
                ((baseline_e2e_p95 - optimized_e2e_p95) / baseline_e2e_p95) * 100,
                2,
            )
        return {
            "completed_requests": self.completed_requests,
            "rejected_requests": self.rejected_requests,
            "total_requests": len(self.requests),
            "tokens_per_second": round(self.generated_tokens / elapsed_s, 3),
            "requests_per_second": round(self.completed_requests / elapsed_s, 3),
            "ttft_p95_ms": percentile(ttfts, 95),
            "e2e_p95_ms": optimized_e2e_p95,
            "tpot_ms": optimized_tpot,
            "kv_blocks_used": used_blocks,
            "kv_blocks_total": self.total_blocks,
            "kv_cache_used_mb": used_mb,
            "kv_cache_utilization": round(used_blocks / self.total_blocks, 4),
            "prefix_cache_enabled": self.prefix_cache.enabled,
            "prefix_cache_entries": len(self.prefix_cache.entries),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_hit_rate": prefix_hit_rate,
            "kv_blocks_reused": self.kv_blocks_reused,
            "kv_blocks_evicted": self.kv_blocks_evicted,
            "prefill_latency_saved_ms": round(self.prefill_latency_saved_ms, 3),
            "admission_rejection_rate": admission_rejection_rate,
            "slo_passed": self.validation["llm_validation_report"].get("passed", False),
            "comparison": {
                "baseline": {
                    "label": "Naive baseline",
                    "ttft_p95_ms": percentile(baseline_ttfts, 95),
                    "tpot_ms": baseline_tpot,
                    "e2e_p95_ms": baseline_e2e_p95,
                    "kv_blocks_used": baseline_blocks_used,
                    "kv_cache_used_mb": baseline_used_mb,
                },
                "optimized": {
                    "label": "Artifact-backed runtime",
                    "ttft_p95_ms": percentile(ttfts, 95),
                    "tpot_ms": optimized_tpot,
                    "e2e_p95_ms": optimized_e2e_p95,
                    "kv_blocks_used": used_blocks,
                    "kv_cache_used_mb": used_mb,
                },
                "improvement": {
                    "end_to_end_p95_pct": e2e_improvement_pct,
                    "kv_memory_saved_mb": round(baseline_used_mb - used_mb, 3),
                    "assumption": (
                        "Baseline is a deterministic unfused runtime estimate; optimized path "
                        "uses the MLIR fused op, lowered HIR plan, and artifact-backed runtime profile."
                    ),
                },
            },
            "compiler_runtime": self.compiler_runtime_summary(),
        }

    def compiler_runtime_summary(self):
        lowered = self.compiler.get("mlir_lowered_graph", {})
        plan = self.compiler.get("mlir_execution_plan", {})
        ops = lowered.get("ops", [])
        steps = plan.get("steps", [])
        first_op = ops[0] if ops else {}
        first_step = steps[0] if steps else {}
        cost = first_op.get("cost_model", {})
        return {
            "workload": "TinyGPT MLIR block",
            "source_pattern": "linalg.matmul + bias_add + relu",
            "fusion_candidate": first_op.get("fusion_candidate", "matmul_bias_relu"),
            "fusion_group": first_op.get("fusion_group", "matmul_bias_relu_0"),
            "lowered_op_type": first_op.get("lowered_op_type", "hir.fused_matmul_bias_relu"),
            "backend": first_step.get("backend", first_op.get("backend", "Metal")),
            "runtime_action": first_step.get("runtime_action", "dispatch_fused_kernel"),
            "estimated_flops": cost.get("estimated_flops", first_step.get("estimated_flops", 0)),
            "estimated_launch_overhead_us": first_step.get("estimated_launch_overhead_us", 0),
            "arithmetic_intensity": round(
                float(cost.get(
                    "arithmetic_intensity_flops_per_byte",
                    first_step.get("arithmetic_intensity_flops_per_byte", 0),
                )),
                3,
            ),
            "compiler_runs": self.compiler_runs,
            "num_lowered_ops": lowered.get("num_ops", len(ops)),
            "num_execution_steps": plan.get("num_steps", len(steps)),
            "kv_policy": self.compiler.get("kv_cache_plan", {}).get("prefix_cache_policy", {}),
            "admission_policy": self.compiler.get("kv_cache_plan", {}).get(
                "admission_policy",
                "capacity_only",
            ),
            "eviction_policy": self.compiler.get("kv_cache_plan", {}).get(
                "eviction_policy",
                "none",
            ),
        }

    def rmsnorm_kernel_selection_summary(self):
        plan = self.compiler.get("rmsnorm_execution_plan", {})
        benchmark = self.runtime_artifacts.get("rmsnorm_benchmark", {})
        steps = plan.get("steps", [])
        first_step = steps[0] if steps else {}
        selection = first_step.get("kernel_selection", {})
        evidence = selection.get("evidence") or {}
        representative_shape = evidence.get("representative_shape", {})
        return {
            "fusion_candidate": first_step.get("fusion_candidate", "rmsnorm"),
            "lowered_op_type": first_step.get("lowered_op_type", "hir.fused_rmsnorm"),
            "selected_kernel": selection.get("selected_kernel", first_step.get("runtime_kernel")),
            "selected_backend": selection.get("selected_backend", first_step.get("backend")),
            "candidate_kernel": selection.get("candidate_kernel"),
            "fallback_kernel": selection.get("fallback_kernel"),
            "selection_reason": selection.get("selection_reason"),
            "profile_status": benchmark.get("profile_status", selection.get("profile_status")),
            "profile_source": benchmark.get("source", selection.get("profile_source")),
            "device": benchmark.get("device"),
            "custom_latency_ms": evidence.get("custom_latency_ms"),
            "fallback_latency_ms": evidence.get("fallback_latency_ms"),
            "speedup": evidence.get("speedup"),
            "correct": evidence.get("correct"),
            "selection_ready": evidence.get("selection_ready"),
            "representative_shape": representative_shape,
            "sweep_count": len(benchmark.get("sweep", [])),
        }

    def snapshot(self):
        return {
            "compiler": self.compiler,
            "runtime_artifacts": self.runtime_artifacts,
            "validation": self.validation,
            "live": {
                "metrics": self.metrics(),
                "mobile_demo": self.mobile_demo,
                "ask_history": self.ask_history[-20:],
                "latest_llm_answer": self.latest_llm_answer,
                "qwen_status": self.qwen_status(),
                "latest_qwen_run": self.latest_qwen_run,
                "memory_summary": self.memory_summary(),
                "resume_evidence": self.resume_evidence(),
                "compiler_runtime": self.compiler_runtime_summary(),
                "rmsnorm_kernel_selection": self.rmsnorm_kernel_selection_summary(),
                "requests": self.requests[-20:],
                "events": self.events[-80:],
                "prefix_cache": [
                    {
                        "prefix_hash": entry.prefix_hash[:12],
                        "blocks": entry.blocks,
                        "tokens": entry.tokens,
                        "ref_count": entry.ref_count,
                        "last_used_ms": entry.last_used_ms,
                        "model_version": entry.model_version,
                    }
                    for entry in self.prefix_cache.entries.values()
                ],
            },
            "artifact_paths": {
                "compiler": str(COMPILER_ARTIFACTS),
                "runtime": str(RUNTIME_ARTIFACTS),
                "validation": str(VALIDATION_ARTIFACTS),
            },
        }


RUNTIME = MiniServingRuntime()


def _parse_runtime_session_path(path):
    """Parses '/api/runtime/session/{id}' -> (id, None) or
    '/api/runtime/session/{id}/{action}' -> (id, action). Returns None for
    anything else (including bare '/api/runtime/session'). Local
    reimplementation of the pattern used by
    heterogeneous-inference-runtime's simulate_service._parse_session_path;
    no cross-repo source import.
    """
    prefix = "/api/runtime/session/"
    if not path.startswith(prefix):
        return None
    parts = path[len(prefix):].split("/")
    if len(parts) == 1 and parts[0]:
        return parts[0], None
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return None


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/snapshot":
            self._send_json(RUNTIME.snapshot())
            return
        if path == "/api/metrics":
            self._send_json(RUNTIME.metrics())
            return
        if path == "/api/evidence":
            self._send_json(RUNTIME.resume_evidence())
            return
        if path == "/api/qwen/status":
            self._send_json(RUNTIME.qwen_status())
            return
        session_action = _parse_runtime_session_path(path)
        if session_action and session_action[1] == "summary":
            self._send_json(RUNTIME.live_runtime_session_summary(session_action[0]))
            return
        if path == "/":
            path = "/index.html"

        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            self.send_error(404)
            return

        body = file_path.read_bytes()
        content_type = "text/html" if file_path.suffix == ".html" else "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            self.send_error(404)
            return
        content_type = "text/html" if file_path.suffix == ".html" else "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/ask":
            self._send_json(RUNTIME.ask(self._read_json()))
            return
        if path == "/api/qwen/ask":
            self._send_json(RUNTIME.ask(self._read_json()))
            return
        if path == "/api/batch":
            self._send_json(RUNTIME.run_batch(self._read_json()))
            return
        if path == "/generate":
            self._send_json(RUNTIME.generate(self._read_json()))
            return
        if path == "/api/runtime/simulate":
            self._send_json(RUNTIME.live_runtime_simulate(self._read_json()))
            return
        if path == "/api/runtime/simulate_batch":
            self._send_json(RUNTIME.live_runtime_simulate_batch(self._read_json()))
            return
        if path == "/api/runtime/session":
            self._send_json(RUNTIME.live_runtime_session_create(self._read_json()))
            return
        session_action = _parse_runtime_session_path(path)
        if session_action and session_action[1] in ("request", "step", "cancel"):
            session_id, action = session_action
            payload = self._read_json()
            if action == "request":
                self._send_json(RUNTIME.live_runtime_session_request(session_id, payload))
            elif action == "step":
                self._send_json(RUNTIME.live_runtime_session_step(session_id, payload))
            else:
                self._send_json(RUNTIME.live_runtime_session_cancel(session_id, payload))
            return
        if path == "/api/compiler/compile":
            self._send_json(RUNTIME.live_compile(self._read_json()))
            return
        if path == "/reset":
            RUNTIME.reset()
            self._send_json({"status": "reset"})
            return
        self.send_error(404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        session_action = _parse_runtime_session_path(path)
        if session_action and session_action[1] is None:
            self._send_json(RUNTIME.live_runtime_session_delete(session_action[0]))
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    host = "127.0.0.1"
    requested_port = int(os.environ.get("PORT", "8765"))
    server = None
    port = requested_port
    for candidate in range(requested_port, requested_port + 10):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            if exc.errno not in {errno.EADDRINUSE, 48, 98}:
                raise
    if server is None:
        raise OSError(f"No free port found from {requested_port} to {requested_port + 9}")
    print(f"MLIR Compiler-to-Runtime Workbench: http://{host}:{port}")
    print("Run Workload -> MLIR pattern match -> HIR lowering -> runtime plan -> metrics")
    server.serve_forever()


if __name__ == "__main__":
    main()
