# Architecture

## Purpose

This repository is a compact demo workbench for an LLM compiler/runtime serving story. It exposes a browser dashboard and JSON API that connect user prompts, optional live Qwen inference, compiler artifacts, runtime artifacts, validation artifacts, and deterministic simulator state.

The project is intended as a handoff/demo surface rather than a production serving stack. It demonstrates what has been implemented, what is artifact-backed, and what remains simulated or reserved for future work.

## Repository Shape

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

## Implemented Components

### `server.py`

`server.py` contains the complete backend:

- HTTP server based on `ThreadingHTTPServer` and `BaseHTTPRequestHandler`.
- Static file serving for `static/index.html`.
- Artifact loading from `artifacts/compiler`, `artifacts/runtime`, and `artifacts/validation`.
- Deterministic runtime simulation for `/generate` and `/api/batch`.
- Optional live Qwen path for `/ask` and `/api/qwen/ask`.
- Metrics aggregation and dashboard snapshot generation.

Important classes and functions:

- `PrefixCacheEntry`: dataclass describing a retained prefix-cache entry.
- `PrefixCacheManager`: deterministic prefix-cache block allocator, lookup, insert, release, and LRU eviction manager.
- `QwenRuntimeAdapter`: optional live HuggingFace/PyTorch Qwen adapter.
- `MiniServingRuntime`: central application state, simulator, artifact reader, metrics builder, and LLM orchestration layer.
- `Handler`: HTTP routing layer.

### `static/index.html`

The dashboard is a single HTML file with embedded CSS and JavaScript. It renders:

- LLM prompt workbench.
- CV placeholder tab.
- BASEMODEL vs optimized Qwen comparison.
- Compiler/runtime pipeline evidence.
- KV cache and memory evidence.
- Runtime event tables.
- Artifact snapshots and validation summaries.

The frontend calls backend JSON endpoints and renders the returned snapshot. It does not contain a separate build system.

### `artifacts/`

Artifacts are committed snapshots copied from external projects:

- `artifacts/compiler`: compiler, MLIR, HIR, execution plan, KV plan, memory plan, scheduling plan, and provenance snapshots.
- `artifacts/runtime`: runtime profile, scheduler, KV pressure, prefill/decode, kernel, distributed, load-balancing, cold-start, and framework comparison snapshots.
- `artifacts/validation`: correctness, SLO, memory, scheduler, KV, framework, distributed, cold-start, and technology-gate validation snapshots.

The server can be pointed at alternate artifact directories with:

- `COMPILER_ARTIFACTS`
- `RUNTIME_ARTIFACTS`
- `VALIDATION_ARTIFACTS`
- `MOBILE_DEMO_SCENARIOS`

## Implemented vs Simulated Behavior

### Real / Implemented

- Python HTTP server and JSON API.
- Static dashboard rendering.
- Artifact loading and snapshot composition.
- Deterministic prefix-cache state, including block allocation, prefix hit/miss accounting, retained prefix blocks, and LRU eviction.
- Optional live HuggingFace Qwen model loading when `torch` and `transformers` are installed and model files are available.
- Live PyTorch prefill call with `use_cache=True`.
- Live token-by-token greedy decode using `past_key_values`.
- Live metrics for prompt tokens, generated tokens, prefill latency, TTFT, TPOT, total latency, tokens/sec, token ids, token fragments, and decode-step latency when Qwen runs successfully.

### Artifact-Backed

- MLIR/HIR/compiler plan evidence.
- KV cache plan and memory plan.
- Scheduler policy comparisons.
- Runtime profile, prefill/decode benchmark, RMSNorm kernel report, page-prefetch report, distributed-serving report, and related runtime evidence.
- Validation reports and SLO summaries.

These are committed snapshots, not recomputed by this repository.

### Simulated / Deterministic Demo

- `/generate` produces deterministic simulated request results from token counts and artifact-derived defaults.
- `/api/batch` runs multiple deterministic simulator requests and joins the live simulator metrics with artifact comparisons.
- Prefix-cache latency savings are estimated by formulas in `MiniServingRuntime.generate`.
- Baseline latency and baseline KV block counts are deterministic estimates, not independently executed baseline serving runs.
- Fallback LLM answers are deterministic strings generated when Qwen is unavailable.

### Placeholder / Not Connected

- The CV tab is a UI placeholder only.
- No camera access is requested.
- No CV model runs.
- No media upload or CV backend route exists.
- No production vLLM, SGLang, Triton, or TensorRT-LLM server is launched by this repo.

## API Surface

GET endpoints:

- `/`: serves `static/index.html`.
- `/api/snapshot`: full dashboard state.
- `/api/metrics`: current deterministic simulator metrics.
- `/api/evidence`: resume/evidence summary.
- `/api/qwen/status`: optional Qwen dependency/model status.

POST endpoints:

- `/ask`: same behavior as `/api/qwen/ask`.
- `/api/qwen/ask`: runs deterministic runtime bookkeeping and optional Qwen BASEMODEL/optimized paths.
- `/api/batch`: runs deterministic batch simulation and artifact comparison.
- `/generate`: runs one deterministic simulator request.
- `/reset`: resets in-memory simulator state.

## Assumptions

- Python 3.11 is the intended runtime for handoff.
- The dashboard is run locally at `127.0.0.1`, default port `8765`.
- Optional live Qwen execution requires an environment with compatible `torch`, `transformers`, and model access/cache.
- Artifact numbers are treated as committed evidence from upstream projects.
- Simulator metrics are local demo metrics and should not be presented as production benchmark numbers.
