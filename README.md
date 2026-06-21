# LLM Serving Runtime Workbench

This repository is the interview demo surface for the LLM compiler/runtime
stack. It presents a generic serving workbench: choose a prompt workload, ask a
local Qwen model when available, compare a direct BASEMODEL path against an
optimized compiler/runtime policy, and inspect the artifact-backed production
evidence behind the optimization story.

The demo consumes committed snapshots from three independent projects:

- `ml-graph-compiler-runtime`: MLIR source, compiler annotations, lowered HIR,
  execution plans, KV contract, and memory plan
- `heterogeneous-inference-runtime`: prefill/decode, scheduler, KV pressure,
  memory, kernel, and serving-policy evidence
- `Inference-Validation-Platform`: correctness, SLO, memory-budget, scheduler,
  and validation reports

The main path is:

```text
Prompt workload
  -> BASEMODEL: full prompt, max_new_tokens=180, direct Qwen decode
  -> Optimized: compact prompt contract, compiler plan, runtime policy
  -> PyTorch prefill/decode with past_key_values
  -> optimized answer text plus BASEMODEL delta
  -> artifact-backed KV/scheduler/memory evidence
  -> validation report
```

There is also a neutral future input slot labeled `Live camera / CV detection
slot`. It is not connected in this demo.

## Project Layout

```text
.
├── server.py
├── static/
│   └── index.html
└── artifacts/
    ├── mobile_demo_scenarios.json
    ├── compiler/
    ├── runtime/
    └── validation/
```

## Run

The artifact dashboard and deterministic fallback use only the Python standard
library:

```bash
python3 server.py
```

Open:

```text
http://127.0.0.1:8765
```

For the live Qwen path, run the server with an environment that has `torch`,
`transformers`, and `accelerate`. The companion runtime project venv can be used
directly:

```bash
cd /Users/allen/Documents/Codex/project/mini-llm-serving-runtime-demo
/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/.venv/bin/python server.py
```

The default model is
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Environment overrides:

```bash
HF_QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
QWEN_DEVICE=auto \
QWEN_MAX_NEW_TOKENS=96 \
BASEMODEL_MAX_NEW_TOKENS=180 \
python3 server.py
```

When Qwen dependencies or model files are unavailable, `/ask` and
`/api/qwen/ask` return an explicit deterministic fallback status. The fallback
is only an offline demo path, not fake live evidence.

## Workloads

The demo uses functional prompt workloads:

- `long_context_summary`: long-context summarization and follow-up extraction
- `instruction_rewrite`: production instruction rewrite
- `technical_explanation`: compiler/runtime optimization explanation

Each workload has:

- `context_items`: prompt facts or notes
- `input_metadata`: prompt shape, audience, priority, and expected style
- `task_context`: title and short task description
- `default_question`: the prompt shown in the text box

## API

Ask the Qwen compiler/runtime path:

```bash
curl -X POST http://127.0.0.1:8765/api/qwen/ask \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "long_context_summary",
    "question": "Summarize the incident and list the highest-risk follow-up items.",
    "llm_mode": "combined"
  }'
```

The response includes:

- `base_model.live_qwen_metrics`
- `optimized.live_qwen_metrics`
- `improvement`
- `decode_steps` and `runtime_trace`
- explicit fallback status when the live path is unavailable

Run one runtime artifact simulator request:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt_tokens": 1024, "max_output_tokens": 128}'
```

Read dashboard state:

```bash
curl http://127.0.0.1:8765/api/snapshot
curl http://127.0.0.1:8765/api/qwen/status
curl http://127.0.0.1:8765/api/metrics
```

Reset runtime state:

```bash
curl -X POST http://127.0.0.1:8765/reset
```

## What The Dashboard Shows

- Input Context: generic prompt workload, context chips, metadata, and a neutral
  future live camera/CV slot
- Ask LLM: sends the selected workload through BASEMODEL and optimized Qwen
  paths
- LLM Answer + Serving Effect: optimized answer, source status, TTFT, total
  latency, TPOT, tokens/sec, prefix-cache state, SLO, and correctness
- Memory Effect: live deterministic prefix-cache state plus committed memory and
  KV evidence
- BASEMODEL vs Qwen Compiler Runtime: direct full-prompt Qwen versus compact
  compiler/runtime policy
- Decode traces: completed token-by-token traces for BASEMODEL and optimized
  paths
- Compiler/runtime/validation explorer: MLIR, HIR, execution plans, KV plans,
  memory timeline, scheduler evidence, and validation reports

## BASEMODEL Comparison

`BASEMODEL` is a real Qwen direct path when optional dependencies and model files
are available:

```text
full prompt
max_new_tokens = 180
no prompt lowering
no chunked prefill
no pressure-aware policy
```

The optimized path uses the same Qwen model with:

```text
compact prompt contract
compiler-generated serving plan from Qwen config
runtime prefill/decode loop
artifact-backed scheduler, KV, memory, and policy evidence
```

The comparison reports TTFT, prefill latency, TPOT, total latency, prompt tokens,
generated tokens, tokens/sec, max token budget, and policy deltas.

## Compiler/Runtime Evidence

The compiler artifacts demonstrate:

- MLIR pattern fusion for `linalg.matmul + bias_add + relu`
- Fusion metadata emitted into annotated MLIR
- MLIR-to-HIR lowering into `hir.fused_matmul_bias_relu`
- Runtime execution-plan generation with backend dispatch contract
- Qwen config conversion into an in-memory serving plan for the live path
- KV cache contract, prefix-cache policy, memory plan, and scheduling plan

The runtime artifacts demonstrate:

- Prefill/decode timing evidence
- Scheduler and admission-policy behavior
- KV pressure and memory footprint reports
- Runtime-aware RMSNorm kernel selection:
  `llm.rmsnorm -> hir.fused_rmsnorm -> fused_rmsnorm_cuda`
- Page-prefetch guardrails, load balancing, cold start, and serving-framework
  comparison artifacts

The validation artifacts demonstrate:

- Correctness and SLO checks
- Memory-budget validation
- Runtime decision validation
- Serving-framework validation
- Scheduler/KV/policy validation reports

## Truth Boundary

Real when available:

- HuggingFace `AutoTokenizer` and `AutoModelForCausalLM`
- PyTorch module execution
- One prefill call with `use_cache=True`
- Token-by-token greedy decode using PyTorch `past_key_values`
- Prompt tokens, generated tokens, prefill latency, TTFT, TPOT, total latency,
  tokens/sec, token ids, token fragments, and decode-step latency

Artifact-backed:

- Compiler MLIR/HIR/plan evidence
- Scheduler, KV pressure, memory, and policy evidence
- Validation and SLO reports

Not claimed:

- Full custom-kernel lowering of every Qwen operator
- Full Qwen weight lowering through the compiler
- Live KV block telemetry from PyTorch internals
- Production vLLM/SGLang/Triton/TensorRT-LLM internals
- Connected live camera/CV inference

## Local Qwen Smoke-Test Results

Recorded on 2026-06-20 using
`heterogeneous-inference-runtime/.venv/bin/python` with cached
`Qwen/Qwen2.5-0.5B-Instruct` weights. Each row runs both BASEMODEL and optimized
paths through `/api/qwen/ask`.

| Workload | BASEMODEL TTFT | Optimized TTFT | BASEMODEL total | Optimized total | Prompt tokens | Generated tokens | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Long Context | 331.506 ms | 62.976 ms | 998.644 ms | 1363.754 ms | 320 -> 163 | 46 -> 88 | Prompt tokens reduced; total latency grew because output length increased |
| Instruction Rewrite | 92.359 ms | 99.774 ms | 255.530 ms | 1320.946 ms | 313 -> 155 | 12 -> 88 | Prompt tokens reduced; optimized answer generated more detail |
| Technical Explain | 93.506 ms | 62.082 ms | 507.964 ms | 986.309 ms | 328 -> 167 | 29 -> 67 | TTFT and prompt size improved; total depends on output length |

Interpretation: the stable demo signal is prompt-token reduction plus the live
BASEMODEL/optimized timing trace. Total latency must be read together with
generated-token count because the two policies can stop at different lengths.

## Refreshing Artifacts

Copy refreshed exports from the source projects into:

```text
artifacts/compiler/
artifacts/runtime/
artifacts/validation/
```

For local testing against live artifact directories:

```bash
export COMPILER_ARTIFACTS=/path/to/compiler/artifacts
export RUNTIME_ARTIFACTS=/path/to/runtime/artifacts
export VALIDATION_ARTIFACTS=/path/to/validation/artifacts
python3 server.py
```

## Verification

```bash
python3 -m py_compile server.py
/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/.venv/bin/python server.py
```

Expected result: old product-theme wording is absent, `py_compile` succeeds,
and the dashboard renders with the three generic workloads.
