# Artifact Snapshot

This directory contains committed snapshot artifacts used by the demo.

```text
artifacts/compiler
  Exported by ml-graph-compiler-runtime.

artifacts/runtime
  Exported by heterogeneous-inference-runtime.

artifacts/validation
  Exported by Inference-Validation-Platform.
```

The demo is intentionally artifact-driven. Each source project can evolve
independently, and this repository can be refreshed by copying in newer
snapshots.

To point the demo at live local artifacts instead of committed snapshots, set:

```bash
export COMPILER_ARTIFACTS=/path/to/compiler/artifacts
export RUNTIME_ARTIFACTS=/path/to/runtime/artifacts
export VALIDATION_ARTIFACTS=/path/to/validation/artifacts
```
