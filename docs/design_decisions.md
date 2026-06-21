# Design Decisions

## Single-File Python Backend

The backend is implemented in `server.py` using only the Python standard library for the web server path.

Tradeoffs:

- Pro: simple to run with `python3 server.py`.
- Pro: no framework dependency for the dashboard and deterministic simulator.
- Pro: easy to inspect in an interview or handoff setting.
- Con: `server.py` is large and mixes HTTP routing, artifact loading, runtime simulation, optional model execution, metrics, and response shaping.
- Con: behavior is harder to test in isolation than if split into modules.

Assumption: portability and demo simplicity were more important than production backend structure.

## Artifact-Driven Evidence

The repository commits exported snapshots instead of recomputing compiler/runtime/validation evidence.

Tradeoffs:

- Pro: the demo can run without the upstream compiler, runtime, or validation projects.
- Pro: evidence remains stable for local review.
- Pro: source projects can evolve independently.
- Con: artifact values can become stale.
- Con: provenance depends on the freshness and trustworthiness of copied artifacts.
- Con: this repo cannot independently verify all artifact claims.

Assumption: committed artifacts are valid snapshots from the named upstream systems.

## Deterministic Runtime Simulator

The `/generate` and `/api/batch` flows simulate request admission, KV block allocation, prefix-cache reuse, latency, and baseline comparisons.

Tradeoffs:

- Pro: gives repeatable local behavior without GPUs or model dependencies.
- Pro: makes KV cache and scheduler concepts visible in the UI.
- Pro: supports demos on machines without live model execution.
- Con: simulator metrics are not production benchmark numbers.
- Con: latency formulas are simplified and tied to artifact defaults.
- Con: simulator state is process-local and disappears on restart.

Assumption: simulator metrics should be presented as deterministic estimates unless paired with artifact or live Qwen source status.

## Optional Live Qwen Path

The Qwen adapter loads HuggingFace `AutoTokenizer` and `AutoModelForCausalLM` only when optional dependencies are available.

Tradeoffs:

- Pro: the default artifact dashboard still works with only Python.
- Pro: live model evidence is possible on properly prepared machines.
- Pro: fallback responses make unavailable dependencies explicit.
- Con: no dependency manifest documents the optional live stack.
- Con: model load errors appear at runtime.
- Con: live results vary by device, model cache, environment, and generation length.

Assumption: live Qwen execution is a bonus capability, not required for basic dashboard operation.

## BASEMODEL vs Optimized Policy Comparison

The ask flow runs two policy configurations:

- BASEMODEL: direct Qwen, full prompt, compiler/runtime policy disabled.
- Optimized: compact prompt contract, prompt lowering flag, chunked-prefill flag, pressure-aware policy flag.

Tradeoffs:

- Pro: makes the comparison boundary explicit in API responses.
- Pro: lets the UI show both direct and optimized paths.
- Con: several optimized policy flags are descriptive in this repo; scheduler/KV/chunked-prefill evidence is artifact-backed rather than live PyTorch internals.
- Con: generated token counts can differ, so total latency is not always directly comparable.

Assumption: users will read total latency together with generated-token count and source status.

## Prompt Contract Keeps Input Honest

The prompt contract intentionally sends only the user-provided prompt to the live LLM path. Scenario buttons replace the textarea content; they do not attach hidden context.

Tradeoffs:

- Pro: reduces hidden prompt behavior and supports clear truth boundaries.
- Pro: simpler to reason about in demos.
- Con: scenario metadata is mostly UI/demo metadata rather than live model context.

Assumption: transparent prompt handling is preferable to richer but hidden scenario context.

## Static Dashboard Without Build Tooling

The UI is implemented as one `static/index.html` file with embedded CSS and JavaScript.

Tradeoffs:

- Pro: no frontend install/build step.
- Pro: easy to serve with the Python server.
- Con: the file is large and mixes layout, styles, rendering logic, and API calls.
- Con: no frontend linting or unit testing is configured.

Assumption: the dashboard is a demo interface rather than a long-lived frontend application.

## Truth Boundary in UI and API

The README and runtime responses explicitly distinguish:

- live Qwen behavior
- artifact-backed evidence
- deterministic fallback/simulator behavior
- not-connected CV placeholder behavior

Tradeoffs:

- Pro: reduces risk of overstating demo behavior.
- Pro: makes handoff safer across tools and reviewers.
- Con: documentation and UI copy must stay synchronized as behavior changes.

Assumption: explicit truth boundaries are part of the product value of this repo.
