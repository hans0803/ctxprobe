# RTX 5060 Ti 16GB — Gemma 4 26B-A4B (QAT q4_0)

**English** · [繁體中文](rtx5060ti-16gb-gemma4-26b-a4b.zh-TW.md)

Measured 2026-08-11. Same card and same tool as the
[Qwen3.6-27B run](rtx5060ti-16gb-qwen3.6-27b.md), which makes this the first
independent check on whether the failure thresholds found there belong to
llama.cpp or to that particular model.

Whole search: **2m39s**, eleven boots.

## Hardware and model

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 16311 MiB reported, 15849 MiB allocatable** |
| llama.cpp | `0a50d99`, CUDA build |
| Model | [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf), 14.44 GB |
| Settings | `-ngl 999`, `--parallel 1`, `--cache-type-k/v q8_0`, flash attention on |

Two architectural properties drive everything below:

- **Sliding-window attention.** 30 layers, of which only **5 are full
  attention**; the other 25 use a 1024-token sliding window whose KV cost is
  fixed rather than growing with context.
- **Mixture of experts.** 128 experts, ~4B parameters active per token out of
  26B total.

Why the QAT build: quantisation-aware training, and at 14.44 GB it is the
largest variant that fits. unsloth's `UD-Q4_K_S` (16.49 GB) and `UD-Q4_K_M`
(16.95 GB) both exceed the card before any KV cache exists.

## Context ceiling: 105,984

| Context | Result | Peak VRAM | Filled |
|---|---|---|---|
| 8,192 | PASS | 14363 MiB | 4,038 |
| 69,632 | PASS | 15293 MiB | 4,038 |
| 100,352 | PASS | 15761 MiB | 4,038 |
| 104,192 | PASS | 15821 MiB | 4,038 |
| **105,984** | **PASS** | 15847 MiB | 4,038 |
| 106,240 | PREFILL_OOM | 15845 MiB | died@508 |
| 106,496 | PREFILL_OOM | 15847 MiB | died@508 |
| 107,008 | PREFILL_OOM | 15791 MiB | died@63 |
| 108,032 | PREFILL_OOM | 15807 MiB | died@63 |
| 115,712 | LOAD_FAIL | — | — |
| 131,072 | LOAD_FAIL | — | — |

Validation of the winner at 95% fill: **confirmed, 100,194 tokens through**.

**Three times the context of Qwen3.6-27B on the same card**, from a model of
comparable size. Sliding-window attention is the reason: only 5 of 30 layers
hold KV that grows with context, against 16 of 64 for Qwen3.6.

## The thresholds are not model-specific

This is the result worth the run. Every failure lands on one of the same two
rungs measured on a completely different architecture:

| Rung | Constant | Qwen3.6-27B (dense + linear attn) | Gemma4-26B-A4B (MoE + sliding window) |
|---|---|---|---|
| 64 | `MMQ_DP4A_MAX_BATCH_SIZE` | `died@64` | `died@63` |
| 512 | `n_ubatch` default | `died@512` | `died@508` |

(63 and 508 rather than 64 and 512 because the ladder converges token counts to
within 3% of target through the server's tokenizer.)

Two models sharing nothing architecturally — dense vs MoE, linear attention vs
sliding window, 64 layers vs 30 — fail at exactly the same two batch sizes. The
thresholds belong to the llama.cpp build and the GPU, as reading
`mmq.cu` predicted. They do not need re-deriving per model.

Note also which rung catches which config: 106,240 and 106,496 (just over the
line) survive to 508, while 107,008 and 108,032 (further over) die at 63. How
far past the ceiling a config sits determines how early it dies.

## Speed

| | Prompt | Prefill | Generate |
|---|---|---|---|
| Search ladder | 4,038 tokens | ~3,510 tok/s | ~120 tok/s |
| **Full validation** | **100,194 tokens** | **1,889.81 tok/s** | **50.82 tok/s** |

Quote the validation row: the ladder figures describe a nearly-empty window.

Against Qwen3.6-27B on the same card, at each model's own ceiling:

| | Gemma4-26B-A4B | Qwen3.6-27B |
|---|---|---|
| Max context | **105,984** | 34,816 |
| Prefill (loaded) | **1,890 tok/s** | 860 tok/s |
| Generate (loaded) | **50.8 tok/s** | 24.9 tok/s |
| File size | 14.44 GB | 15.44 GB |

Roughly double the throughput at triple the context. The generation gap is what
MoE buys — ~4B parameters active per token instead of 27B — and the context gap
is what sliding-window attention buys. Neither is a statement about output
quality, which this project does not measure.

Generation holds up better under load too: 120 tok/s on an empty window against
50.8 at 100K, versus Qwen3.6's 26 → 24.9. The proportional drop is far larger
here, which is consistent with 5 full-attention layers still having to attend
over a 100K cache.

## Reproducing

```bash
ctxprobe gemma-4-26B-A4B-it-qat-q4_0.gguf --min 8192
```

`--max` comes from the GGUF metadata (262144, capped to 131072). No
`--reasoning-budget` needed — Gemma 4 is not a thinking model, and no `SILENT`
verdicts appeared.
