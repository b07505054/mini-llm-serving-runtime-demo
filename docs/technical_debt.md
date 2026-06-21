# Technical Debt

## Weak Spots

- `server.py` is large and owns too many responsibilities: HTTP routing, artifact loading, simulation, model execution, metrics, response formatting, and state management.
- `static/index.html` is also large and mixes markup, CSS, API access, state, and rendering.
- The optional Qwen dependency set is documented in README text but not captured in a `requirements.txt`, `pyproject.toml`, or lockfile.
- All runtime state is in memory and process-local.
- Artifact schemas are used directly as dictionaries without typed validation.
- Missing artifact files silently fall back to empty dictionaries or empty strings, which keeps the demo running but can hide broken evidence.
- Endpoint input validation is minimal; most numeric fields are cast directly with `int(...)`.
- JSON parse errors and malformed request bodies are not handled with structured API errors.
- The global `RUNTIME` instance is shared across threaded requests without locks.

## Missing Tests

No test files were found in the repository.

Important areas needing tests:

- `PrefixCacheManager` allocation, release, lookup, insert, and eviction behavior.
- `_prefix_fingerprint` threshold and hashing behavior.
- `/generate` admission, rejection, prefix-hit, prefix-miss, and eviction flows.
- `metrics()` percentile and comparison calculations.
- `/api/qwen/ask` fallback response shape when Qwen is unavailable.
- `QwenRuntimeAdapter.status()` dependency reporting.
- Artifact-loading fallbacks.
- Static file path traversal protection.
- API behavior for malformed JSON and invalid numeric inputs.

## Duplicated or Repeated Logic

- Many artifact keys are loaded manually in `MiniServingRuntime.__init__`.
- Metric rendering in the frontend repeats similar card/table construction patterns.
- Backend response objects repeat policy, source-status, metric, and evidence fields across Qwen/base/optimized packages.
- The UI and README both maintain truth-boundary descriptions; these can drift.

## Unclear Naming

- `mobile_demo` and `mobile_demo_scenarios.json` now describe generic LLM serving workloads rather than a clearly mobile-specific demo.
- `base`, `BASEMODEL`, `base_model`, and `base_qwen_result` refer to related but slightly different concepts.
- `live_synthetic_metrics` in batch output combines current simulator metrics with artifact comparison context.
- `compiler_ready` in Qwen status maps to live Qwen readiness, not an actual compiler process.
- The frontend uses `displayJson(...).replaceAll("wa" + "sted", "unused")`, and the backend uses `prefetch_metric.get("wa" + "sted_prefetch_blocks")`. This appears intended to avoid displaying a specific key name, but it is surprising and should be explained or renamed.

## Future Risks

- Threaded requests can mutate shared prefix-cache and metrics state concurrently.
- Artifact schema changes from upstream projects can break dashboard panels silently.
- Live Qwen behavior can vary substantially by hardware and installed dependency versions.
- BASEMODEL and optimized paths may generate different output lengths, making total latency comparisons easy to misread.
- Missing dependency manifests make environment reproduction harder.
- Large single files increase merge conflict risk and make Claude/Codex handoff more error-prone.
- The UI placeholder for CV could be mistaken for implemented functionality if copy drifts.
- Since normal operation writes no files, state cannot be inspected after process exit unless logs are captured externally.

## Documentation Drift Risks

- The README includes recorded smoke-test values. Those values should remain dated and should not be generalized as benchmark claims.
- Any new artifact refresh should update docs if source projects, schemas, or truth boundaries change.
- If a live serving backend is added later, docs must distinguish it from the current deterministic simulator and artifact-backed evidence.
