# CLAUDE.md

## Project Context

This repository is a local LLM serving runtime demo/workbench. It has a Python HTTP server, a static dashboard, committed compiler/runtime/validation artifacts, a deterministic serving simulator, and an optional live HuggingFace Qwen path.

Clearly distinguish these categories in all changes:

- Implemented: Python server, static dashboard, artifact loading, deterministic prefix-cache simulator, optional HuggingFace/PyTorch Qwen execution.
- Artifact-backed: compiler plans, runtime reports, validation reports, SLOs, scheduler/KV/memory evidence.
- Simulated: `/generate`, `/api/batch`, deterministic fallback answers, estimated baseline comparisons, prefix-cache latency savings.
- Not connected: CV camera slot, production vLLM/SGLang/Triton/TensorRT-LLM internals, live PyTorch KV block telemetry.

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
- At minimum, run `python3 -m py_compile server.py` after Python changes.
- Add focused tests for prefix-cache logic, simulator metrics, artifact fallbacks, and API response shapes when changing those areas.

## Handoff Rules

- Explain changes after implementation.
- State what was tested.
- State what was not tested.
- Preserve truth boundaries in README, docs, UI copy, and API responses.
- Do not present deterministic simulator metrics as production benchmark results.
- Do not present committed artifacts as freshly generated live evidence.

## Common Commands

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
python3 -m py_compile server.py
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
