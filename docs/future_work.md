# Future Work

## Near-Term Improvements

- Add focused unit tests for prefix-cache behavior, simulator admission/rejection, metrics, and Qwen fallback responses.
- Add a small dependency manifest for optional live Qwen execution, ideally separate from the standard-library dashboard path.
- Split `server.py` into focused modules:
  - artifact loading
  - prefix cache
  - deterministic simulator
  - Qwen adapter
  - metrics
  - HTTP handlers
- Add artifact schema validation with clear startup warnings when required files or keys are missing.
- Add structured API error responses for malformed JSON and invalid inputs.
- Add thread-safety around mutable runtime state or switch to a single-request event loop for demo consistency.

## Documentation and Handoff

- Keep truth-boundary language close to the code paths that produce responses.
- Add a short artifact-refresh checklist.
- Document expected optional Qwen environment setup in a reproducible way.
- Add example responses for `/generate`, `/api/qwen/ask`, `/api/snapshot`, and `/api/qwen/status`.
- Keep smoke-test metrics dated and environment-specific.

## Dashboard Improvements

- Split frontend rendering into smaller functions or separate files if the dashboard grows.
- Add frontend smoke tests for rendering with minimal/empty artifacts.
- Make artifact source status visually consistent across panels.
- Ensure any future CV UI remains clearly labeled as not connected until a real backend exists.

## Runtime and Model Work

- Add a real continuous batching backend only if the project scope expands beyond demo simulation.
- Integrate live KV telemetry only if the model runtime exposes reliable internals.
- Add support for configurable generation parameters such as temperature, top-p, and stop criteria.
- Add bounded model-loading behavior and clearer model-cache diagnostics.
- Persist selected run traces to disk only when explicitly requested.

## Artifact Pipeline

- Automate copying refreshed artifacts from upstream projects.
- Include artifact manifests with source commit, generation command, timestamp, and validation status.
- Fail loudly or warn clearly when artifacts are internally inconsistent.
- Add schema-version checks for compiler, runtime, and validation artifacts.

## Quality Gates

- Add `python -m py_compile server.py` as a minimal check.
- Add unit tests for non-trivial logic before larger changes.
- Add a lightweight smoke test that starts the server, fetches `/api/snapshot`, posts `/generate`, and posts `/api/qwen/ask` in fallback mode.
- Consider static analysis once the code is modularized.

## Product Scope Decisions

- Decide whether the repo should remain a demo dashboard or become a reusable local serving workbench.
- If it remains a demo, prioritize clarity, reproducibility, and truth boundaries.
- If it becomes a workbench, prioritize typed modules, tests, dependency management, and stronger API contracts.
