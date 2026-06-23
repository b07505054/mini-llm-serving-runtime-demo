# CLAUDE.md

## Project Context

This repository is a local LLM serving runtime demo/workbench. It has a Python HTTP server, a static dashboard, committed compiler/runtime/validation artifacts, a deterministic serving simulator, and an optional live HuggingFace Qwen path.

Clearly distinguish these categories in all changes:

- Implemented: Python server, static dashboard, artifact loading, deterministic prefix-cache simulator, optional HuggingFace/PyTorch Qwen execution.
- Artifact-backed: compiler plans, runtime reports, validation reports, SLOs, scheduler/KV/memory evidence.
- Simulated: `/generate`, `/api/batch`, deterministic fallback answers, estimated baseline comparisons, prefix-cache latency savings.
- Not connected: CV camera slot, production vLLM/SGLang/Triton/TensorRT-LLM internals, live PyTorch KV block telemetry.

Portfolio demo positioning:

- Keep this as the default interview demo for compiler, runtime, ML infrastructure, LLM serving, validation, memory planning, and systems roles.
- Do not describe PocketChef-AI as replacing this demo. Treat the HTML demo and iPhone app as separate front ends for different audiences.
- This repo is the complete HTML path for explaining compiler/runtime/validation artifacts without depending on CoreML, iPhone availability, Xcode, or live device execution.

High-value artifact-backed evidence includes:

- Distributed serving, load-balancing, fault-tolerance, and gRPC contract reports.
- Cold-start, prefill/decode, page-prefetch, scheduler, KV, and memory validation artifacts.
- Serving-framework comparison artifacts for vLLM-style, SGLang-style, Triton Server-style, and TensorRT-style stories.
- GPU PGO-like RMSNorm and runtime decision validation artifacts imported from the compiler/runtime evidence chain.

Do not invent benchmark numbers. If metrics are estimated, label them estimated.

## Runtime

- Python 3.11.
- Standard-library dashboard path should continue to run with `python3 server.py`.
- Optional live Qwen path may require `torch`, `transformers`, and model files/cache.

## Code Style

- Prefer dataclasses.
- Avoid unnecessary classes.
- Keep functions under 100 lines when practical.
- Use type hints.
- Prefer simple modular design.
- Avoid over-engineering.
- Use composition over inheritance.
- No giant classes.
- Keep source changes small and easy to review.

## Testing

- Write tests for non-trivial logic.
- Run tests after changes.
- At minimum, run `bash scripts/check.sh` after Python changes; it compiles every tracked Python file.
- Add focused tests for prefix-cache logic, simulator metrics, artifact fallbacks, and API response shapes when changing those areas.

## Handoff Rules

- Explain changes after implementation.
- State what was tested.
- State what was not tested.
- Preserve truth boundaries in README, docs, UI copy, and API responses.
- Do not present deterministic simulator metrics as production benchmark results.
- Do not present committed artifacts as freshly generated live evidence.

## Common Commands

Run the canonical validation command used by CI
(`.github/workflows/check.yml`):

```bash
bash scripts/check.sh
```

Run the demo:

```bash
python3 server.py
```

Open:

```text
http://127.0.0.1:8765
```

Check syntax:

```bash
bash scripts/check.sh
```

Example simulator request:

```bash
curl -X POST http://127.0.0.1:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt_tokens": 1024, "max_output_tokens": 128}'
```

Example Qwen/fallback request:

```bash
curl -X POST http://127.0.0.1:8765/api/qwen/ask \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "long_context_summary", "question": "Summarize this prompt.", "llm_mode": "combined"}'
```

Check optional Qwen dependency/model status:

```bash
curl http://127.0.0.1:8765/api/qwen/status
```

Live Qwen metrics are real only when the response reports live Qwen execution. Compiler, scheduler, KV, memory, and serving-framework panels remain artifact-backed even during live Qwen runs.

## Portfolio-Level Policy

When this repository is maintained inside the `systems-portfolio` wrapper, follow the root `CLAUDE.md` for shared documentation hierarchy, benchmark honesty, demo-selection guidance, and Git authorship rules. Keep this file focused on repository-specific capabilities, truth boundaries, and validation commands.
