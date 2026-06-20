# HTML LLM Serving Runtime Demo Platform

This is the main interview demo surface for the compiler/runtime and LLM serving
systems work. It replaces the fragile iPhone/CoreML live-demo path with a stable
HTML experience: a PocketChef-style phone scenario asks a deterministic local
food assistant, while the page shows KV-cache behavior, TTFT/TPOT, throughput,
SLO validation, and the compiler/runtime artifacts behind the result.

The demo does not merge the source projects. Instead, it consumes committed
artifact snapshots from three independent systems:

- `ml-graph-compiler-runtime`: MLIR source, annotated MLIR, lowered HIR JSON, execution plans
- `heterogeneous-inference-runtime`: runtime profile and workload timing artifacts
- `Inference-Validation-Platform`: correctness, SLO, scheduler, and validation reports

The main story is:

```text
Mock phone food snapshot
  -> ingredient/nutrition/question context
  -> deterministic LLM answer path
  -> TinyGPT MLIR block
  -> MLIR fusion pass
  -> annotated MLIR
  -> lowered HIR JSON
  -> runtime execution plan
  -> KV prefix-cache admission policy
  -> heterogeneous runtime metrics
  -> validation report
```

The interactive phone flow is deterministic so it can run during an interview
without iPhone hardware, CoreML model bundles, Xcode signing, Ollama, a GPU, or
network access. PocketChef-AI remains an optional mobile product shell; this repo
is the primary HTML demonstration platform.

## Project Layout

```text
.
├── server.py
├── static/
│   └── index.html
└── artifacts/
    ├── mobile_demo_scenarios.json
    ├── compiler/
    │   ├── tiny_gpt_serving.mlir
    │   ├── mlir_fused_graph.mlir
    │   ├── mlir_lowered_graph.json
    │   └── mlir_execution_plan.json
    ├── runtime/
    └── validation/
```

## Run

No third-party Python dependencies are required.

```bash
python3 server.py
```

Open:

```text
http://127.0.0.1:8765
```

## API

Ask the phone-style LLM assistant:

```bash
curl -X POST http://127.0.0.1:8765/ask \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "breakfast_bowl",
    "ingredients": ["egg", "rice", "spinach"],
    "nutrition": {"calories": 520, "protein_g": 28},
    "question": "How can I make this higher protein?",
    "llm_mode": "combined"
  }'
```

Run one workload instance:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt_tokens": 1024, "max_output_tokens": 128}'
```

Read the full dashboard snapshot:

```bash
curl http://127.0.0.1:8765/api/snapshot
```

Read live metrics:

```bash
curl http://127.0.0.1:8765/api/metrics
```

Reset runtime state:

```bash
curl -X POST http://127.0.0.1:8765/reset
```

## What the Dashboard Shows

- Phone-style HTML demo input: mock camera panel, detected ingredients,
  nutrition/recipe context, LLM question, and deterministic answer
- LLM serving effect: TTFT, TPOT, E2E latency, tokens/sec, prefix-cache hit/miss,
  prefill saved, SLO pass, and correctness pass
- Memory evidence: compiler memory plan, runtime KV footprint, page-prefetch
  memory guard, validation budget, and live prefix-cache memory effect
- MLIR compiler pipeline: source MLIR, fusion pass, annotated MLIR, lowered HIR, runtime plan
- Fusion details: fusion candidate, fusion group, lowered op type, backend, runtime action
- Runtime lowering: `hir.fused_matmul_bias_relu` dispatched to the configured backend
- Runtime-aware kernel selection: `hir.fused_rmsnorm` selects
  `fused_rmsnorm_cuda` over `torch_rmsnorm` only when runtime benchmark evidence
  proves the custom CUDA kernel is faster and correct
- Artifact provenance: compiler version, git commit, pass pipeline, artifact hashes
- Plan comparison: Metal, CPU, and Hybrid candidate plans with estimated and measured metrics
- KV runtime policy: prefix-cache hit rate, reused/evicted KV blocks, admission rejects,
  and prefill latency saved from cached prefixes
- Memory timeline: allocation, reuse, and free events with validation status
- Real backend profiling: HuggingFace LlamaForCausalLM profiling on available PyTorch backends
  with TTFT, TPOT, batch/sequence scaling, and operator bottleneck breakdown
- Baseline vs compiler-lowered runtime comparison: TTFT, TPOT, E2E, KV memory
- Serving-framework comparison artifacts for vLLM/SGLang-style scheduling,
  Triton Server-style dynamic batching/backend routing, and TensorRT-style
  engine/profile dispatch
- Cold-start artifacts showing model load, backend initialization, TensorRT
  engine deserialize/context creation, first-token warmup, and steady-state metrics
- Runtime events: `mlir_pattern_matched`, `lowered_to_hir`, prefill/decode events, completion
- KV policy events: `prefix_cache_hit`, `prefix_cache_miss`, `kv_blocks_evicted`,
  `admission_rejected`, and `prefix_cache_inserted`
- Validation status: correctness pass/fail, SLO status, max logit diff
- Raw artifact snapshots from compiler, runtime, and validation layers

The top of the dashboard is the interview path:

```text
Phone Demo Input -> Ask LLM -> Serving Runtime Metrics -> Validation Evidence
```

The lower sections remain the evidence explorer for compiler, runtime, kernel,
memory, and validation artifacts.

## Memory Evidence

The dashboard treats memory as a first-class serving metric. The visible memory
panels combine committed artifacts with deterministic live prefix-cache state:

```text
Compiler memory plan
  peak_decode_memory_mb = 673
  memory_budget_mb = 8192
  reuse_enabled = true
  fits_memory_budget = true

Runtime serving profile
  peak_memory_mb = 1636.75
  peak_kv_cache_mb = 868.75
  oom_events = 0

KV-cache analysis
  total_blocks = 512
  peak_blocks_used = 278
  block_utilization = 0.543
  fragmentation_ratio = 0.05
  failed_allocations = 0
```

Validation reports the compiler-side memory budget result:

```text
budget_utilization = 0.0822
allocations = 4
reuse_events = 1
frees = 3
issues = []
```

The live HTML demo also reports current prefix-cache blocks, live KV MB, reused
blocks, evictions, and prefill latency saved after repeated prompts. Those live
values come from the deterministic in-process prefix-cache simulator. They are
not real iPhone/CoreML memory measurements.

## Current Compiler/Runtime Evidence

The committed artifacts currently demonstrate:

- MLIR pattern fusion for `linalg.matmul + bias_add + relu`
- Fusion metadata emitted into annotated MLIR:
  `fusion.candidate`, `fusion.group`, and `fusion.role`
- MLIR-to-HIR lowering into `hir.fused_matmul_bias_relu`
- Backend placement for the lowered op, currently targeting `Metal`
- Runtime execution-plan generation with `dispatch_fused_kernel`
- Runtime-aware RMSNorm kernel selection:
  `llm.rmsnorm -> hir.fused_rmsnorm -> fused_rmsnorm_cuda`
  using measured PyTorch-vs-custom-CUDA benchmark evidence from
  `heterogeneous-inference-runtime`
- Lightweight cost metadata: estimated FLOPs, memory traffic, arithmetic intensity,
  and launch overhead
- TinyGPT-style planning artifacts: prefill/decode phase split, KV-cache block
  plan, prefix-cache/admission policy, memory budget, and scheduling contract

The demo consumes `kv_cache_plan.json` from `ml-graph-compiler-runtime` and
enforces the KV policy at request time. Requests compute a prefix hash, reuse
resident prefix KV blocks on cache hit, evict finished prefixes with LRU when
capacity is tight, and reject admission only after eviction cannot free enough
blocks.

The strongest compiler claim in this snapshot is the MLIR fusion-to-runtime
bridge:

```text
linalg.matmul + bias add + ReLU
  -> fusion annotation
  -> HIR fused op
  -> runtime execution plan
  -> backend dispatch contract
```

The RMSNorm path demonstrates the runtime-aware compiler loop:

```text
llm.rmsnorm
  -> hir.fused_rmsnorm
  -> PyTorch RMSNorm vs custom CUDA RMSNorm benchmark
  -> compiler selects fused_rmsnorm_cuda
```

The committed benchmark snapshot on an NVIDIA GTX 1650 Max-Q records
`fused_rmsnorm_cuda` at `0.02975 ms` versus `torch_rmsnorm` at `0.086751 ms`
for the representative shape, with correctness passing and a `2.916x` speedup.

The LLM-shaped workload is used to exercise this compiler/runtime path. It is
not presented as a full LLM serving framework.

## Production-Oriented Artifacts

This snapshot also includes three production-style artifact groups:

- `artifact_provenance.json`: compiler version, git commit, pass pipeline, and
  SHA-256 hashes for emitted compiler artifacts
- `candidate_execution_plans.json` and `plan_benchmark_results.json`: candidate
  Metal, CPU, and Hybrid plans with compiler estimates and runtime benchmark
  results
- `memory_timeline.json` and `memory_validation_report.json`: allocation,
  reuse, free events, peak memory, budget utilization, and validation status
- `real_llama_profile.json`: real PyTorch backend execution profile for a
  HuggingFace LlamaForCausalLM model, including MPS/CPU availability,
  TTFT/TPOT across batch and sequence shapes, and per-operator bottleneck
  breakdown

These artifacts make the demo auditable: the dashboard can show which compiler
produced the plan, which backend plan was selected, and how memory was allocated
and reused over the workload. Real backend profiling is separated from the
deterministic demo estimates so the dashboard can distinguish measured backend
evidence from simulated serving-path estimates.

## Baseline Comparison

The dashboard includes a deterministic comparison between:

- `Unfused baseline`: a simple runtime estimate without MLIR fusion/lowering
- `MLIR-lowered runtime`: the demo path using the committed compiler, runtime,
  and validation artifacts

This is not a claim that a real model has been optimized by those exact
percentages. It is a demo mechanism for showing how the same workload is
reported before and after MLIR fusion, HIR lowering, runtime planning, and
artifact-backed validation are introduced.

## Refreshing Artifacts

The committed snapshots live under `artifacts/`. To refresh them from the
source projects, copy new exports into:

```text
artifacts/compiler/
artifacts/runtime/
artifacts/validation/
```

Current source artifact locations:

```text
/Users/allen/Desktop/project/ml-graph-compiler-runtime/integration_bundle/apple_demo_artifacts
/Users/allen/Desktop/project/heterogeneous-inference-runtime/results/llm_runtime_artifacts
/Users/allen/Desktop/project/inference-validation-platform/integration_artifacts
```

Example refresh:

```bash
cp /Users/allen/Desktop/project/ml-graph-compiler-runtime/integration_bundle/apple_demo_artifacts/* artifacts/compiler/
cp /Users/allen/Desktop/project/heterogeneous-inference-runtime/results/llm_runtime_artifacts/* artifacts/runtime/
cp /Users/allen/Desktop/project/inference-validation-platform/integration_artifacts/* artifacts/validation/
```

For local testing against live artifact directories, set environment variables:

```bash
export COMPILER_ARTIFACTS=/path/to/compiler/artifacts
export RUNTIME_ARTIFACTS=/path/to/runtime/artifacts
export VALIDATION_ARTIFACTS=/path/to/validation/artifacts
python3 server.py
```

## Demo Narrative

This repository turns three infrastructure projects into a visible phone-to-LLM
serving demo. The HTML phone scenario provides the product-shaped input. The
compiler emits MLIR and runtime-facing HIR artifacts. The runtime consumes the
lowered plan and produces timing/profile artifacts. The validation platform
turns runtime results into correctness and SLO reports. The demo shell ties them
together as an optimization workbench that is stable enough for interviews.

## What Is Real Versus Simulated

- Real: committed MLIR/HIR/compiler artifacts from `ml-graph-compiler-runtime`
- Real: committed runtime and profiling artifacts from
  `heterogeneous-inference-runtime`
- Real: HuggingFace `LlamaForCausalLM` execution on available PyTorch backends
  in `real_llama_profile.json`
- Real: validation artifact snapshots from `Inference-Validation-Platform`
- Simulated: the interactive `/ask` phone demo answer is deterministic and uses
  committed scenario inputs from `artifacts/mobile_demo_scenarios.json`
- Simulated: the interactive `/generate` endpoint uses deterministic timing
  formulas so the dashboard remains portable without requiring a GPU or model
  download at demo time
- Not claimed: real iPhone/CoreML inference, production vLLM/SGLang/Triton/
  TensorRT-LLM serving, or live framework-internal modification

This split is intentional: the dashboard is a presentation layer over committed
compiler/runtime/validation evidence, while the live controls make that evidence
easy to inspect during an interview.
