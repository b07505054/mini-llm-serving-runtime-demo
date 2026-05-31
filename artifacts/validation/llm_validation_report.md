# LLM Runtime Validation Report: llm-runtime-demo-001

**Result:** PASS
**Latency budget:** p95 decode <= 20.0000 ms
**p95 decode latency:** 15.9000 ms
**Correctness passed:** `True`
**Max logit diff:** 0.000800
**Peak memory:** 1240.0 MB

## SLO Report

| Metric | Value |
|---|---:|
| SLO passed | `True` |
| TTFT p95 | 412.8000 ms |
| TPOT p95 | 15.9000 ms |
| E2E p95 | 1170.2000 ms |
| Queue wait p95 | 38.1000 ms |
| SLO violation rate | 0.0470 |
| Admission rejection rate | 0.0470 |
| Tokens/sec | 84.7000 |
| Requests/sec | 2.8000 |

## Scheduler Analysis

| Metric | Value |
|---|---:|
| Avg queue wait | 12.4000 ms |
| p95 queue wait | 38.1000 ms |
| Max active requests | 8 |
| Decode batch efficiency | 0.8200 |

## KV Cache Analysis

| Metric | Value |
|---|---:|
| Peak blocks used | 812 |
| Block utilization | 0.7900 |
| Fragmentation ratio | 0.0800 |
| Evictions | 0 |
| Failed allocations | 0 |

## Request Timeline

| Request | Arrival | Prefill start | Decode start | Finish | Status |
|---|---:|---:|---:|---:|---|
| `req-001` | 0.0 ms | 2.0 ms | 190.0 ms | 820.0 ms | `completed` |
| `req-002` | 6.0 ms | 10.0 ms | 198.0 ms | 835.0 ms | `completed` |
| `req-003` | 10.0 ms | 16.0 ms | 205.0 ms | 850.0 ms | `completed` |
| `req-004` | 18.0 ms | 26.0 ms | 211.0 ms | 862.0 ms | `completed` |
| `req-005` | 24.0 ms | 34.0 ms | 219.0 ms | 879.0 ms | `completed` |
| `req-006` | 31.0 ms | 43.0 ms | 227.0 ms | 891.0 ms | `completed` |
| `req-007` | 40.0 ms | 59.1 ms | 238.0 ms | 910.0 ms | `completed` |
| `req-008` | 52.0 ms | 90.1 ms | 252.0 ms | 928.0 ms | `completed` |

## Summary

Validation platform turns runtime traces into correctness, latency, memory, and scheduling reports.
