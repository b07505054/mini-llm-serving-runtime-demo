# ml-graph-compiler-runtime

Compiler emits LLM execution, memory, scheduling, validation, and KV-cache
planning artifacts for an LLM serving/runtime demo.

## Generate artifacts

```bash
python3 src/ml_graph_compiler_runtime/generate_llm_artifacts.py \
  --config configs/tiny_gpt_llm_config.json \
  --out artifacts/apple_demo
```

The Apple-side demo should consume the JSON files under `artifacts/apple_demo/`.

## Output files

- `llm_graph_ir.json`
- `serving_execution_plan.json`
- `kv_cache_plan.json`
- `memory_plan.json`
- `scheduling_plan.json`
- `validation_manifest.json`

