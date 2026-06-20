# Runtime Artifact Validation Report: runtime-artifact-validation-001

**Result:** PASS
**Source runtime:** `/Users/allen/Documents/Codex/project/heterogeneous-inference-runtime/results/llm_runtime_artifacts`
**Model:** `tiny-gpt`

## Prefill / Decode

- Prefill latency: `38.316` ms
- p95 decode latency: `2.511` ms
- Tokens/sec: `1470.548`

## SLO

- p95 end-to-end latency: `1406.597` ms
- p95 queue wait: `1128.618` ms
- OOM events: `0`
- Admission rejection rate: `0.0`

## KV Cache

- Peak blocks used: `278` / `512`
- Block utilization: `0.543`
- Fragmentation ratio: `0.05`
- Peak KV cache: `868.75` MB

## Scheduler

- Policy: `cost_aware_memory_pressure_page_prefetch`
- Decode batch events: `5`
- Avg decode batch size: `6.4`
- p95 queue wait: `1128.618` ms

## Runtime Decision Validation

- Selected policy: `cost_aware_memory_pressure_page_prefetch`
- Decision validation passed: `True`
- Tokens/sec delta: `1172.501`
- p95 latency delta: `-6179.235` ms
- Decode batch efficiency delta: `0.675`
- Pressure-limited candidates: `20`
- Regression detected: `False`

## Serving Framework Targets

- Selected style: `vllm_sglang_style`
- Validation passed: `True`
- Available styles: `['baseline_fcfs', 'tensorrt_style', 'triton_server_style', 'vllm_sglang_style', 'vllm_style_page_prefetch']`
- TTFT: `38.316` ms
- TPOT p95: `2.511` ms/token
- Throughput: `1470.548` tokens/s
- Peak KV cache: `868.75` MB
- Selection reason: `page prefetch candidate improved TPOT/e2e/throughput without KV regression`

## Cold Start / Initialization

- Validation passed: `True`
- Cold TTFT: `98.316` ms
- Warm TTFT: `38.316` ms
- First request penalty: `60.0` ms
- Steady-state TPOT p95: `2.511` ms/token
- Available artifacts: `['onnx_fp32', 'tensorrt_fp16_engine', 'tensorrt_int8_engine', 'executorch_xnnpack_pte']`
- TensorRT available: `False`

## Framework Trace Adapters

- vLLM validation passed: `True`
  input: `vllm style synthetic request/decode trace`
  decision: `continuous batching admission and paged-KV pressure cap`
  metric: `TTFT, TPOT, throughput, queue wait, KV pressure, decode batch efficiency`
  throughput: `1470.548` tokens/s
- SGLang validation passed: `True`
  input: `sglang style synthetic request/decode trace`
  decision: `request/decode scheduling with prefix-reuse metadata`
  metric: `TTFT, TPOT, throughput, queue wait, KV pressure, decode batch efficiency`
  throughput: `1470.548` tokens/s

## Technology Gate

- Validation passed: `True`
- Main-plan technologies: `11`
- Recorded backlog technologies: `7`
- Invalid main-plan items: `[]`

## vLLM-Style Page Prefetch

- Validation passed: `True`
- Input: `vLLM-style request/decode trace plus allocated KV block map and pending decode candidates`
- Decision: `prefetch next decode KV pages only when memory pressure is below the prefetch budget`
- Metric: `prefetch hit rate, wasted prefetch blocks, TPOT p95, decode p95, throughput, queue wait, OOM/rejection rate`
- Selected policy: `cost_aware_memory_pressure_page_prefetch`
- Hit rate: `0.8808`
- TPOT p95 delta: `-0.7331` ms/token
- Tokens/sec delta: `234.406`

## Distributed Serving

- Validation passed: `True`
- Selected policy: `least_queue`
- Cache-aware check: `True`
- Selection reason: `least-queue retained because KV-aware routing regressed TPOT or throughput`

## Load Balancing

- Validation passed: `True`
- Selected policy: `least_queue`
- Cache hit delta: `0.2188`
- TPOT p95 delta: `1.423` ms/token
- Throughput delta: `-141.8577` tokens/s

## Worker Health / Failover

- Validation passed: `True`
- Retry count: `1`
- Failover count: `1`
- Quarantine count: `1`
- Failed requests: `0`
- TTFT p95 regression: `-100.9461` ms

## Protobuf Contract

- Validation passed: `True`
- Service defined: `True`
- Stub generation: `skipped_tool_not_installed`
- Claim boundary: `protobuf contract only; no production gRPC deployment is claimed`

## GPU PGO-like Feedback

- Validation passed: `True`
- Input: `compiler-emitted HIR RMSNorm op plus runtime shape/workload distribution`
- Decision: `profile-guided kernel selection among CUDA/Triton/PyTorch candidates by shape bucket`
- Metric: `kernel p95 latency, effective bandwidth, TPOT projection, throughput projection`
- Selected kernel: `fused_rmsnorm_cuda`
- Representative shape: `16x4096:fp32`
- TPOT delta: `0.061856` ms/token

## Backend Placement

- Heterogeneous execution detected: `True`
- Backend counts: `{'gpu': 5, 'cpu': 5}`
- Op counts: `{'attention_prefill': 5, 'kv_cache_update': 5}`

## Validation Positioning

This report validates runtime artifacts produced by `heterogeneous-inference-runtime` rather than only simulating worker behavior inside the validation platform.
