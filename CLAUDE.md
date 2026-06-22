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

## Documentation Hierarchy

Truth must flow in the following order:

Code
↓
Artifacts
↓
README.md
↓
CLAUDE.md
↓
docs/

Lower levels must never contradict higher levels.

Documentation must describe reality rather than invent behavior.

If uncertainty exists, trust code and generated artifacts.

Never exaggerate capabilities.

Never claim production behavior unless code and artifacts support it.

## README Contract

README.md exists to answer:

1. What is it?
2. Why is it interesting?
3. How do I run it?
4. What results does it produce?

README should emphasize user-facing understanding.

Avoid implementation details unless necessary.

Avoid maintenance instructions.

## CLAUDE.md Contract

CLAUDE.md exists to answer:

1. How do I maintain it?
2. What commands are canonical?
3. Which components are implemented?
4. Which components are simulated?
5. Which validation commands must pass?
6. What files should not be changed casually?

CLAUDE.md is intended for maintainers and future AI agents.

## docs/ Contract

docs/ exists to answer:

1. Why is it designed this way?
2. What tradeoffs were made?
3. What is measured versus modeled?
4. What assumptions exist?
5. What limitations remain?
6. What future work is possible?

docs/ explains architecture and rationale rather than usage.

## Documentation Principles

Code > Artifacts > README > CLAUDE.md > docs/

Never reverse this order.

Never infer unsupported features.

Never create claims unsupported by code or artifacts.

Prefer conservative wording.

Call synthetic benchmarks synthetic.

Call simulated systems simulated.

Distinguish measured behavior from modeled behavior.

## Git Authorship Policy

The user is the sole maintainer and owner of this repository.

AI agents may modify files as requested.

AI agents must not add AI authorship metadata.

Never add:

* Co-Authored-By entries
* Co-authored-by trailers
* Claude authorship metadata
* AI signatures
* Generated-by-AI footers
* any metadata that makes an AI system appear as a repository contributor

Commit policy:

* By default, do not run git commit.
* If the user explicitly asks in the current conversation to commit, an AI agent may run git add and git commit.
* Commits created by an AI agent must use the user's configured git author and committer identity.
* Commit messages must not mention AI authorship unless the user explicitly asks.
* Before committing, show git status and the staged diff summary when practical.

Push policy:

* By default, do not run git push.
* Only run git push if the user explicitly asks in the current conversation.
* Never force-push unless the user explicitly asks for a force push and the reason is explained.

History policy:

* Do not create branches, rewrite history, rebase, reset, or amend commits unless the user explicitly asks in the current conversation.
* Never rewrite public history without explicit user approval.

Ownership rule:

* The user remains the sole author/maintainer for portfolio presentation purposes.
* No AI system should appear as a repository contributor.
