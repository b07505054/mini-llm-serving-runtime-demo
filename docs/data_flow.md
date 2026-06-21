# Data Flow

## Inputs

Primary inputs:

- User prompt text from the dashboard textarea or `/api/qwen/ask` JSON body.
- Scenario id and LLM mode from `artifacts/mobile_demo_scenarios.json` or request payload.
- Token-count inputs for deterministic simulator endpoints: `prompt_tokens`, `max_output_tokens`, optional `prefix`, optional `prefix_tokens`.
- Committed compiler/runtime/validation artifact files.
- Optional environment variables for model and artifact locations.

Environment inputs:

- `PORT`
- `ARTIFACT_ROOT`
- `COMPILER_ARTIFACTS`
- `RUNTIME_ARTIFACTS`
- `VALIDATION_ARTIFACTS`
- `MOBILE_DEMO_SCENARIOS`
- `HF_QWEN_MODEL`
- `QWEN_DEVICE`
- `QWEN_MAX_NEW_TOKENS`
- `BASEMODEL_MAX_NEW_TOKENS`

## Server Startup Flow

1. `server.py` resolves repository paths and artifact directories.
2. `MiniServingRuntime` loads compiler, runtime, validation, and scenario artifacts.
3. `MiniServingRuntime.reset()` initializes in-memory simulator state:
   - prefix-cache manager
   - event list
   - request history
   - counters
   - latest answer/run snapshots
4. `ThreadingHTTPServer` binds to `127.0.0.1`, trying ports from `PORT` through `PORT + 9`.
5. The browser loads `static/index.html`.
6. The frontend calls `/api/snapshot` and renders the dashboard.

## `/generate` Flow

1. Request JSON supplies `prompt_tokens`, `max_output_tokens`, optional `prefix`, optional `prefix_tokens`, and optional `request_id`.
2. The server computes total required KV tokens and required KV blocks from `block_size_tokens`.
3. If prefix caching is enabled and the prompt is long enough, `_prefix_fingerprint` hashes model version, prefix label, and prefix token count.
4. `PrefixCacheManager.lookup` returns a retained prefix entry on hit.
5. The runtime emits deterministic events:
   - request arrival
   - prefix hit or miss
   - MLIR pattern matched
   - HIR lowering
   - KV eviction if needed
   - admission or rejection
   - prefill start/end
   - decode start/end
   - request finished
6. Capacity checks either reject the request or allocate new KV blocks.
7. Latency values are calculated from artifact defaults and deterministic formulas.
8. Cacheable prefix blocks may be retained; non-retained blocks are released.
9. A result row is appended to request history.

Important note: this flow is a deterministic simulator. It does not execute a model.

## `/api/qwen/ask` Flow

1. The request selects a scenario and LLM mode.
2. The raw prompt is converted into a prompt contract:
   - fixed system rules
   - no hidden context
   - no metadata
   - no task context
   - `question` set to the submitted prompt
3. Prompt/output token counts are estimated unless explicitly supplied.
4. `MiniServingRuntime.generate()` runs first to update deterministic simulator state.
5. Two Qwen policies are built:
   - `BASEMODEL`: full prompt, no prompt lowering, no chunked prefill, no pressure-aware policy, larger token budget.
   - `Optimized Compiler Runtime`: compact prompt mode, prompt lowering enabled, chunked prefill flag enabled, pressure-aware policy flag enabled.
6. `QwenRuntimeAdapter.ask()` runs for both policies if optional dependencies and model are available.
7. If Qwen is unavailable, the response includes deterministic fallback answer text and explicit unavailable status.
8. The response combines:
   - simulator result
   - optimized answer or fallback answer
   - BASEMODEL package
   - optimized package
   - improvement calculation
   - validation summary
   - memory summary
   - compiler plan
   - runtime/decode traces
   - evidence source paths

Important note: live Qwen metrics are real only when `source_status` is `qwen_live`. Scheduler, KV, memory, and compiler evidence remain artifact-backed even during live Qwen runs.

## Optional Qwen Flow

When dependencies and model files are available:

1. `QwenRuntimeAdapter.status(load_model=True)` imports `torch` and `transformers`.
2. Device selection uses configured `QWEN_DEVICE` or auto-selects CUDA, MPS, then CPU.
3. The tokenizer and `AutoModelForCausalLM` are loaded.
4. The prompt contract is formatted with `apply_chat_template` when available.
5. A prefill forward pass runs with `use_cache=True`.
6. Greedy decode loops one token at a time using `past_key_values`.
7. Per-token decode steps and latencies are recorded.
8. The adapter returns answer text, metrics, trace, and a compiler-plan summary derived from the live model config.

## Outputs

Backend outputs:

- JSON API responses.
- In-memory request/event history.
- In-memory prefix-cache entries.
- In-memory latest LLM answer and latest Qwen run snapshots.

Frontend outputs:

- Dashboard panels for Qwen answer/comparison.
- Compiler/runtime pipeline views.
- Memory and KV-cache views.
- Runtime event and request tables.
- Raw artifact snapshots.

No files are written during normal server operation.

## Metrics

Live Qwen metrics when available:

- `prompt_tokens`
- `generated_tokens`
- `prefill_ms`
- `ttft_ms`
- `tpot_ms`
- `total_latency_ms`
- `tokens_per_second`
- `decode_steps`

Deterministic simulator metrics:

- completed/rejected/total requests
- tokens/sec and requests/sec based on simulator elapsed time
- TTFT p95 and E2E p95 from simulated requests
- TPOT from artifact `avg_decode_latency_ms`
- KV blocks used/total
- KV cache MB used
- prefix-cache hits, misses, hit rate, reused blocks, evicted blocks
- admission rejection rate
- estimated baseline/optimized comparison

Artifact-backed metrics:

- runtime profile latency, memory, throughput, OOM, rejection counts
- prefill/decode benchmark values
- scheduler policy comparisons
- page-prefetch metrics
- RMSNorm kernel selection metrics
- SLO and validation report fields

Estimated metrics are generated by formulas in `server.py` and should be labeled as estimated in any external presentation.

## Important Data Structures

- `PrefixCacheEntry`: prefix hash, retained blocks, token count, ref count, last-used time, model version.
- `PrefixCacheManager.entries`: active deterministic prefix-cache entries.
- `MiniServingRuntime.compiler`: loaded compiler artifacts.
- `MiniServingRuntime.runtime_artifacts`: loaded runtime artifacts.
- `MiniServingRuntime.validation`: loaded validation artifacts.
- `MiniServingRuntime.requests`: recent deterministic simulator request results.
- `MiniServingRuntime.events`: recent deterministic runtime events.
- `MiniServingRuntime.ask_history`: recent LLM ask responses.
- `live.metrics`: dashboard metrics produced by `MiniServingRuntime.metrics()`.
- `live.memory_summary`: combined artifact and deterministic live prefix-cache memory view.
- `live.resume_evidence`: curated claim-to-artifact evidence summary.
