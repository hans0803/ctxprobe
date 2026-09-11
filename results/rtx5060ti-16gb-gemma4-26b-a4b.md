# RTX 5060 Ti 16GB — Gemma 4 26B-A4B (QAT q4_0)

**English** · [繁體中文](rtx5060ti-16gb-gemma4-26b-a4b.zh-TW.md)

Measured 2026-08-11; the 8GB section re-measured 2026-09-11. Same card and same tool as the
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

## Spilling to an 8GB card

Constrained to a real 8GB card's budget (7799 MiB free, see
[spill-cost.md](../docs/spill-cost.md) for why a 16GB 5060 Ti can stand in for
the 8GB one):

| Method | Fits? | Generate |
|---|---|---|
| **`--cpu-moe`** (experts in RAM) | **yes — 2.4 GB of VRAM** | **39.7 tok/s** |
| `--n-cpu-moe 15` (half the layers) | no | — |
| `-ngl 20` (dense-style, control) | no | — |

**`-ngl` is the wrong tool for a MoE model.** Every layer contains experts, so
cutting layers doesn't remove the bulk. Moving experts by tensor type does:
weights drop from 14.44 GB to about 2.4 GB.

With that much VRAM freed, the 8GB card runs the model's **full 262,144
context** — that is Gemma 4's trained limit, not a memory edge:

| Context | Result | Peak VRAM | Model's own | Prefill | Generate | Filled |
|---|---|---|---|---|---|---|
| 131,072 | PASS | 12,168 MiB | 4,117 MiB | 700 tok/s | 38.6 tok/s | 4,038 |
| 262,144 | PASS | 14,168 MiB | 6,117 MiB | 702 tok/s | 39.0 tok/s | 4,038 |
| **262,144** | **PASS at 95% fill** | 14,168 MiB | 6,117 MiB | **501 tok/s** | **17.95 tok/s** | **249,013** |

"Model's own" subtracts the 8,051 MiB the holding process and its CUDA context
occupied. At the full window the model uses 6.1 GB of the 7.8 GB budget, so
there is 1.7 GB to spare when the model itself runs out of context. A 26B
model, on 8GB, at 256K — with room. The 249K-token prompt took 497 s, inside
ctxprobe's 900 s floor, so the time budget never came into it.

Generate at a full window is 17.95 tok/s against 39 on an empty one. That is
the same shape as the 16GB run's 120 → 50.8: five full-attention layers still
attend over the whole cache while the other 25 stop at their sliding window.
KV grows 2,000 MiB per 131,072 tokens here — 15.3 MiB per 1K, a sixth of what
the same 30 layers would need with global attention in every one of them.

The August run reported 131,072 for this configuration. That was ctxprobe's
search cap at the time, not a measurement of the card; the cap is now 262,144
and this section is the re-measurement.

### Why it isn't slower

At 39.7 tok/s with experts in system RAM, the obvious question is how a 4B-active
model sustains that across DDR4. The routing config answers it:

```
expert_count       = 128
expert_used_count  = 8      <- top-8 routing
expert_ff_length   = 704
```

Each expert is `3 × 2816 × 704 = 5.95M` parameters (gate, up, down); at q4_0's
18 bytes per 32 weights that is 3.19 MiB. Per token the model activates
30 layers × 8 experts = 240 of them:

```
240 × 3.19 MiB   = 0.80 GB per token
39.7 tok/s × 0.80 GB = 31.9 GB/s
```

Measured DDR4 bandwidth on this host: 43.1 GB/s copy (read+write), 27.0 GB/s
single-threaded read. So 31.9 GB/s is close to saturation — **it is bandwidth
bound**, exactly as expected.

The reason it feels faster than "4B active over DDR4" predicts is that only
**1.43B of those 4B are experts**. The other ~2.6B — attention, shared layers,
embeddings — never leave the GPU. MoE's advantage here isn't only that few
parameters are active; it's that the parameters which *are* sparse happen to be
the ones you can evict.

## Reproducing

```bash
ctxprobe gemma-4-26B-A4B-it-qat-q4_0.gguf --min 8192
```

`--max` comes from the GGUF metadata (262144). The 8GB section adds
`python tools/hold-vram.py 7800 &` first and `-- --cpu-moe` at the end. No
`--reasoning-budget` needed — Gemma 4 is not a thinking model, and no `SILENT`
verdicts appeared.
