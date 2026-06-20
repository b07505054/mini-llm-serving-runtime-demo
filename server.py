#!/usr/bin/env python3
from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
import uuid
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
                "artifact_type": "mobile_demo_scenarios",
                "truth_boundary": "Deterministic HTML demo input, not real iPhone/CoreML inference.",
                "scenarios": [],
            },
        )
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

    def _scenario_by_id(self, scenario_id):
        for scenario in self.mobile_demo.get("scenarios", []):
            if scenario.get("id") == scenario_id:
                return scenario
        scenarios = self.mobile_demo.get("scenarios", [])
        return scenarios[0] if scenarios else {}

    def _normalize_ingredients(self, payload, scenario):
        ingredients = payload.get("ingredients")
        if not ingredients:
            ingredients = scenario.get("ingredients", [])
        normalized = []
        for item in ingredients:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                confidence = float(item.get("confidence", 0.86))
            else:
                name = str(item).strip()
                confidence = 0.86
            if name:
                normalized.append({"name": name, "confidence": round(confidence, 3)})
        return normalized

    def _prompt_contract(self, payload, scenario, ingredients, question, llm_mode):
        nutrition = payload.get("nutrition") or scenario.get("nutrition", {})
        recipe = payload.get("recipe") or scenario.get("recipe", {})
        ingredient_names = [item["name"] for item in ingredients]
        return {
            "source": "html_mobile_demo",
            "truth_boundary": "PocketChef-style deterministic input; not real iPhone/CoreML inference.",
            "llm_mode": llm_mode,
            "system_rules": [
                "Use only detected ingredients as visual facts.",
                "Mark assumptions clearly.",
                "Keep the answer concise and practical.",
                "No medical claims.",
            ],
            "visual_facts": ingredient_names,
            "nutrition_estimate": nutrition,
            "recipe_context": recipe,
            "question": question,
        }

    def _estimate_prompt_tokens(self, prompt_contract):
        visual = len(prompt_contract.get("visual_facts", []))
        question_words = len(prompt_contract.get("question", "").split())
        mode = prompt_contract.get("llm_mode", "combined")
        mode_adjustment = {
            "base": 120,
            "runtime": 96,
            "compiler": 80,
            "combined": 72,
        }.get(mode, 96)
        return max(128, min(2048, mode_adjustment + visual * 34 + question_words * 7))

    def _estimate_output_tokens(self, question, llm_mode):
        base = 112 if len(question.split()) > 8 else 88
        if llm_mode == "compiler":
            base -= 16
        if llm_mode == "combined":
            base -= 24
        return max(48, min(160, base))

    def _deterministic_answer(self, prompt_contract, scenario):
        ingredients = prompt_contract.get("visual_facts", [])
        nutrition = prompt_contract.get("nutrition_estimate", {})
        question = prompt_contract.get("question", "")
        mode = prompt_contract.get("llm_mode", "combined")
        names = ", ".join(ingredients[:4]) or "the detected ingredients"
        protein = nutrition.get("protein_g") or nutrition.get("protein") or "unknown"
        calories = nutrition.get("calories", "unknown")
        recipe = prompt_contract.get("recipe_context", {})
        title = recipe.get("title") or scenario.get("answer_seed", "quick bowl")

        lowered = "I would keep the visual facts as"
        if mode in {"compiler", "combined"}:
            lowered = "After prompt lowering, the visual facts are"
        runtime_note = "The runtime path reuses the shared context when this scenario repeats."
        if mode == "base":
            runtime_note = "This is the baseline deterministic answer path."
        if mode == "runtime":
            runtime_note = "The runtime policy keeps the serving context warm for lower TTFT."

        if "protein" in question.lower():
            advice = (
                f"Add a second egg, tofu, Greek yogurt sauce, or canned tuna if it fits the meal. "
                f"The current estimate is about {protein}g protein at {calories} kcal."
            )
        elif "waste" in question.lower() or "leftover" in question.lower():
            advice = (
                "Turn the detected items into a bowl, wrap, or fried rice. Use the softest greens first "
                "and keep sauce separate so leftovers survive one more meal."
            )
        elif "cook" in question.lower() or "make" in question.lower():
            advice = (
                f"Make {title}: warm the base, add {names}, season with salt, acid, and a little fat, "
                "then finish with a crunchy topping if available."
            )
        else:
            advice = (
                f"{title} is the safest plan from {names}. Add seasoning, balance protein and fiber, "
                "and treat anything outside the detections as an assumption."
            )

        return f"{lowered}: {names}. {advice} {runtime_note}"

    def ask(self, payload):
        scenario_id = payload.get("scenario_id") or "breakfast_bowl"
        scenario = self._scenario_by_id(scenario_id)
        llm_mode = payload.get("llm_mode") or scenario.get("default_mode") or "combined"
        question = (payload.get("question") or scenario.get("default_question") or "What can I make from this?").strip()
        ingredients = self._normalize_ingredients(payload, scenario)
        prompt_contract = self._prompt_contract(
            payload,
            scenario,
            ingredients,
            question,
            llm_mode,
        )
        prompt_tokens = int(payload.get("prompt_tokens") or self._estimate_prompt_tokens(prompt_contract))
        output_tokens = int(payload.get("max_output_tokens") or self._estimate_output_tokens(question, llm_mode))
        prefix = "|".join(
            [
                "mobile-demo",
                scenario_id,
                llm_mode,
                ",".join(item["name"].lower() for item in ingredients),
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
        answer = self._deterministic_answer(prompt_contract, scenario)
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
            "mode": llm_mode,
            "scenario_id": scenario_id,
            "ingredients": ingredients,
            "nutrition": prompt_contract.get("nutrition_estimate", {}),
            "question": question,
            "prompt_contract": prompt_contract,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
            "tokens_per_second": self.runtime_artifacts["runtime_profile"].get(
                "tokens_per_second",
                self.metrics().get("tokens_per_second", 0),
            ),
            "validation": validation,
            "memory": memory,
            "evidence": {
                "runtime_profile": "artifacts/runtime/runtime_profile.json",
                "prefill_decode": "artifacts/runtime/prefill_decode_benchmark.json",
                "validation_report": "artifacts/validation/llm_validation_report.json",
                "slo_report": "artifacts/validation/slo_report.json",
            },
        }
        self.latest_llm_answer = response
        self.ask_history.append(response)
        self.ask_history = self.ask_history[-20:]
        self._event(
            "mobile_llm_answer_ready",
            request_id,
            scenario_id=scenario_id,
            mode=llm_mode,
            prefix_cache=response.get("prefix_cache"),
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
                "with deterministic live prefix-cache state; no real iPhone/CoreML memory is claimed."
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
                "fragmentation_ratio": kv_analysis.get("fragmentation_ratio"),
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
                "wasted_prefetch_blocks": prefetch_metric.get("wasted_prefetch_blocks"),
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
                "memory_summary": self.memory_summary(),
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
        if path == "/generate":
            self._send_json(RUNTIME.generate(self._read_json()))
            return
        if path == "/reset":
            RUNTIME.reset()
            self._send_json({"status": "reset"})
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MLIR Compiler-to-Runtime Workbench: http://{host}:{port}")
    print("Run Workload -> MLIR pattern match -> HIR lowering -> runtime plan -> metrics")
    server.serve_forever()


if __name__ == "__main__":
    main()
