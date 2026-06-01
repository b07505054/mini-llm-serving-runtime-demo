# MLIR Compiler-to-Runtime Optimization Workbench

This is a self-contained demo project for showing a compiler/runtime optimization
pipeline. It does not merge the source projects. Instead, it consumes committed
artifact snapshots from three independent systems:

- `ml-graph-compiler-runtime`: MLIR source, annotated MLIR, lowered HIR JSON, execution plans
- `heterogeneous-inference-runtime`: runtime profile and workload timing artifacts
- `Inference-Validation-Platform`: correctness, SLO, scheduler, and validation reports

The main story is:

```text
TinyGPT MLIR block
  -> MLIR fusion pass
  -> annotated MLIR
  -> lowered HIR JSON
  -> runtime execution plan
  -> heterogeneous runtime metrics
  -> validation report
```

The workload is LLM-shaped and is used to exercise compiler transformation,
runtime lowering, backend dispatch, and validation.

## Project Layout

```text
.
├── server.py
├── static/
│   └── index.html
└── artifacts/
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

- MLIR compiler pipeline: source MLIR, fusion pass, annotated MLIR, lowered HIR, runtime plan
- Fusion details: fusion candidate, fusion group, lowered op type, backend, runtime action
- Runtime lowering: `hir.fused_matmul_bias_relu` dispatched to the configured backend
- Artifact provenance: compiler version, git commit, pass pipeline, artifact hashes
- Plan comparison: Metal, CPU, and Hybrid candidate plans with estimated and measured metrics
- Memory timeline: allocation, reuse, and free events with validation status
- Baseline vs compiler-lowered runtime comparison: TTFT, TPOT, E2E, KV memory
- Runtime events: `mlir_pattern_matched`, `lowered_to_hir`, prefill/decode events, completion
- Validation status: correctness pass/fail, SLO status, max logit diff
- Raw artifact snapshots from compiler, runtime, and validation layers

## Current Compiler/Runtime Evidence

The committed artifacts currently demonstrate:

- MLIR pattern fusion for `linalg.matmul + bias_add + relu`
- Fusion metadata emitted into annotated MLIR:
  `fusion.candidate`, `fusion.group`, and `fusion.role`
- MLIR-to-HIR lowering into `hir.fused_matmul_bias_relu`
- Backend placement for the lowered op, currently targeting `Metal`
- Runtime execution-plan generation with `dispatch_fused_kernel`
- Lightweight cost metadata: estimated FLOPs, memory traffic, arithmetic intensity,
  and launch overhead
- TinyGPT-style planning artifacts: prefill/decode phase split, KV-cache block
  plan, memory budget, and scheduling contract

The strongest compiler claim in this snapshot is the MLIR fusion-to-runtime
bridge:

```text
linalg.matmul + bias add + ReLU
  -> fusion annotation
  -> HIR fused op
  -> runtime execution plan
  -> backend dispatch contract
```

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

These artifacts make the demo auditable: the dashboard can show which compiler
produced the plan, which backend plan was selected, and how memory was allocated
and reused over the workload.

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

For local testing against live artifact directories, set environment variables:

```bash
export COMPILER_ARTIFACTS=/path/to/compiler/artifacts
export RUNTIME_ARTIFACTS=/path/to/runtime/artifacts
export VALIDATION_ARTIFACTS=/path/to/validation/artifacts
python3 server.py
```

## Demo Narrative

This repository turns three infrastructure projects into a visible compiler and
runtime demo. The compiler emits MLIR and runtime-facing HIR artifacts. The
runtime consumes the lowered plan and produces timing/profile artifacts. The
validation platform turns runtime results into correctness and SLO reports. The
demo shell ties them together as an optimization workbench.
