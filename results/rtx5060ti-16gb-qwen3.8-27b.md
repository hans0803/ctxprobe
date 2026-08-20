# RTX 5060 Ti 16GB — Qwen3.8-27B

**English** · [繁體中文](rtx5060ti-16gb-qwen3.8-27b.zh-TW.md)

Measured 2026-08-20, on the same card that produced the
[Qwen3.6-27B numbers](rtx5060ti-16gb-qwen3.6-27b.md). Same architecture, same
tool, same method — which makes this a controlled comparison rather than two
unrelated benchmarks.

The headline looked like a doubling. It wasn't.

## Hardware

Identical to the Qwen3.6 run, deliberately.

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 16311 MiB reported, 15849 MiB allocatable** |
| Driver | 595.84 |
| llama.cpp | `0a50d99` (2026-07-23), CUDA build |
| Host | i7-14700, 64 GB DDR4 — not a factor in any number below |

Everything runs entirely in VRAM (`-ngl 999`, no CPU offload).

## The model is architecturally identical to Qwen3.6

Both report `model_type: qwen3_5` and both produce GGUFs whose
`general.architecture` is `qwen35`. Field by field:

```
num_hidden_layers        64      full_attention_interval  4
num_attention_heads      24      num_key_value_heads      4
head_dim                 256     hidden_size              5120
vocab_size               248320  max_position_embeddings  262144
```

No llama.cpp update was needed — the existing build already had `qwen35`.

One difference, and it matters later: **Qwen3.8's GGUF carries a 65th block.**

```
qwen35.block_count            = 65     (Qwen3.6: 64)
qwen35.nextn_predict_layers   = 1      (Qwen3.6: absent)
```

`blk.64` is a full transformer block — `attn_q`/`attn_k`/`attn_v`,
`ffn_up`/`gate`/`down`, plus `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm` —
totalling **0.425 B parameters**. It is the multi-token-prediction head.

It is loaded unconditionally. `src/models/qwen35.cpp` only makes the trunk
optional for a *dedicated* MTP file:

```c
const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
```

This GGUF has `blk.0.attn_norm.weight`, so `mtp_only` is false and both the
trunk and `blk.64` load as required tensors. The MTP *graph*, however, is a
separate entry point that plain `llama-server` decoding never calls.

**Cost: about 350 MiB of weights for a block that does not execute.** It does
*not* cost KV cache — see the slope measurements below, which is where an
earlier guess of ours died.

## The quant label does not fix the bits

Qwen3.8's GGUF repo has no plain `IQ4_XS`, only `UD-IQ4_XS`. Treating that as
the successor to Qwen3.6's `IQ4_XS` is the obvious move and it is wrong.

Bits-per-weight computed from ggml's actual block sizes, not estimated:

| | Params | Tensor bytes | **bpw** |
|---|---|---|---|
| Qwen3.6 `IQ4_XS` | 26,895,998,464 | 14,714 MiB | **4.5892** |
| Qwen3.8 `UD-IQ4_XS` | 27,320,697,856 | 13,582 MiB | **4.1703** |
| Qwen3.8 `UD-Q4_K_S` | 27,320,697,856 | 14,636 MiB | **4.4939** |

Qwen3.8 has *more* parameters (the MTP block) in a *smaller* file, because
`UD-IQ4_XS` is 0.42 bpw leaner than the thing it appears to replace. The two
files are packed on completely different principles:

| | Qwen3.6 `IQ4_XS` | Qwen3.8 `UD-IQ4_XS` |
|---|---|---|
| Distinct quant types | 4 | **12** |
| Bulk | IQ4_XS 20.2 B | IQ4_XS 13.6 B + IQ3_S 3.8 B + Q3_K 1.9 B |
| Below 3 bits | none | 0.80 B params |
| `token_embd` | Q4_K | **Q3_K** |
| `output` | Q6_K | **Q5_K** |

So the comparison that means something is against `UD-Q4_K_S` at 4.4939 bpw —
within 2.07% of Qwen3.6's 4.5892.

## Context ceilings

Four full binary searches, each winner re-validated against a prompt filling
95% of its window. `--parallel 1`, `-ngl 999`, `--reasoning-budget 0`.

| Model | bpw | KV | **Ceiling** | Peak VRAM | Prefill | Generate | Filled |
|---|---|---|---|---|---|---|---|
| `UD-IQ4_XS` | 4.1703 | `q8_0` | **61,952** | 15847 | 739.93 | 23.39 | 58,735 |
| `UD-IQ4_XS` | 4.1703 | `f16` | **36,352** | 15835 | 825.34 | 27.04 | 34,331 |
| `UD-Q4_K_S` | 4.4939 | `q8_0` | **34,304** | 15847 | 820.15 | 24.98 | 32,333 |
| `UD-Q4_K_S` | 4.4939 | `f16` | **19,712** | 15833 | 875.09 | 27.23 | 18,681 |

Prefill and generate are measured at the fill shown, so they are not comparable
across rows — a deeper window is slower to prefill and slower to decode into.

## At equal precision, Qwen3.8 gets slightly less context

| KV | Qwen3.6 `IQ4_XS` (4.5892) | Qwen3.8 `UD-Q4_K_S` (4.4939) | Δ |
|---|---|---|---|
| `q8_0` | 34,816 | **34,304** | **−512** |
| `f16` | 20,224 | **19,712** | **−512** |

Both modes land exactly 512 tokens — two search steps — below Qwen3.6, despite
Qwen3.8 being the *slightly* lower-precision file of the two.

**The apparent 61,952 vs 34,816 (+78%) is entirely an artefact of quantisation
packaging.** Nothing about Qwen3.8 the model extends the context you can run on
this card. Take the same bits per weight and it is marginally behind.

The residual is fixed allocation, not per-token cost. Backing the KV out of the
f16 ceilings gives a fixed footprint of 14,559 MiB for Qwen3.6 against 14,582
MiB for Qwen3.8 — 23 MiB apart, even though Qwen3.8's tensors are 78 MiB
*smaller* on disk. That is ~100 MiB of non-weight allocation this llama.cpp
build does not itemise, so it is reported rather than explained.

## KV cache: what quantising it actually buys

Peak VRAM against context is startlingly linear — these are differences between
adjacent PASS rows, not a fitted line:

| | Measured slope | 16-layer theory | Excess |
|---|---|---|---|
| `f16` | **63.48 MiB/1K** | 64.0 | ~0 |
| `q8_0` | **38.09 MiB/1K** | 34.0 | **+4.1** |

The f16 slope is identical across `UD-IQ4_XS`, `UD-Q4_K_S`, **and Qwen3.6**
(recomputed from its own published table: 63.48). That is what rules out the
MTP block having a KV cache — a 17th attention layer would put f16 at 68.0.

Theory is `n_full_attn_layers × n_kv_heads × head_dim × 2 (K+V) × bytes`:
16 × 4 × 256 × 2 = 32,768 elements per token, times 2 bytes at f16 = 64.0
MiB/1K, times 1.0625 bytes at q8_0 = 34.0 MiB/1K.

**f16 matches theory. q8_0 overshoots by 4.1 MiB/1K.** That excess is 4,198
bytes per token, and one layer's per-token KV held at f16 is
4 × 256 × 2 × 2 = 4,096 bytes — a 2.5% match. The shape of it is a
dequantisation scratch buffer sized for one layer at a time, growing linearly
with context. f16 needs no such buffer.

The practical consequence:

| | Per 1K tokens | Ceiling gain over f16 |
|---|---|---|
| `UD-IQ4_XS` | 63.48 → 38.09 | 36,352 → 61,952 (**+70.4%**) |
| `UD-Q4_K_S` | 63.5 → 37.8 | 19,712 → 34,304 (**+74.0%**) |
| Qwen3.6 `IQ4_XS` | 63.48 → ~38 | 20,224 → 34,816 (+72.1%) |

**q8_0 saves 40% of per-token KV, not the 47% the type sizes imply.** The gap is
the scratch buffer. It is still overwhelmingly worth it, but a calculator that
divides KV bytes by two will overestimate the context you get.

## The prompt ladder earned its third rung

Three of the four searches failed the way this card always has — through the
512-token rung, the first full `n_ubatch`:

| Config | Died at | Rung |
|---|---|---|
| `UD-IQ4_XS` q8_0 62,208 | `died@509` | `n_ubatch` |
| `UD-IQ4_XS` f16 36,608 | `died@509` | `n_ubatch` |
| `UD-Q4_K_S` q8_0 34,560 | `died@509` | `n_ubatch` |
| **`UD-Q4_K_S` f16 19,968** | **`died@4024`** | **`n_batch`** |

The [Qwen3.6 page](rtx5060ti-16gb-qwen3.6-27b.md) noted that every failure
measured to that point landed on one of the first two rungs, leaving 4096 as a
rung that cleared `n_batch` (2048) in principle but had never actually caught
anything. **19,968 is the counter-example.** It survived 64 tokens, survived
512, and died on 4,024.

A ladder that stopped at 512 would have called 19,968 a PASS.

Worth noting the 512-rung failures span both KV types, and therefore both
flash-attention states — `q8_0` forces flash attention on, `f16` leaves it at
`auto`. The failure point does not move, which supports the existing attribution
to ggml's memory pool growing against batch shape rather than to the attention
implementation.

## Practical configuration

Ceilings are ceilings. Leave headroom:

```bash
llama-server -m Qwen3.8-27B-UD-IQ4_XS.gguf \
  -ngl 999 -c 57344 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --reasoning-budget 0
```

57,344 sits ~4,600 tokens under the measured 61,952, which is roughly the
`n_ubatch` compute-buffer margin that separated PASS from PREFILL_OOM.

Which file to pull depends on what you are optimising:

| Want | File | Get |
|---|---|---|
| Most context | `UD-IQ4_XS` (4.17 bpw) | 61,952 |
| Most precision that still fits | `UD-Q4_K_S` (4.49 bpw) | 34,304 |
| Parity with Qwen3.6 | `UD-Q4_K_S` | 34,304 vs 34,816 |

`UD-Q4_K_M` (16.46 GB) does not fit — it leaves ~150 MiB for KV.

## Reproducing

```bash
ctxprobe Qwen3.8-27B-UD-IQ4_XS.gguf  --min 32768 --max 98304 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-Q4_K_S.gguf  --min  8192 --max 45056 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-IQ4_XS.gguf --kv f16 --min 8192 --max 40960 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-Q4_K_S.gguf --kv f16 --min 4096 --max 28672 -- --reasoning-budget 0
```

Model: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

## What this run added to the method

1. **Check bits, not names.** Two files labelled as the same Q4 class differed
   by 0.42 bpw. Any cross-model context comparison that skips this is measuring
   the quantiser, not the model. See [model-quant.md](../docs/model-quant.md).
2. **Quantised KV carries a context-scaling overhead.** 38.09 MiB/1K measured
   against 34.0 theoretical — so q8_0 saves 40% of the per-token cost, not the
   50% its type size implies. This refines
   [kv-cache-quant.md](../docs/kv-cache-quant.md), which currently describes
   q8_0 as "half the memory"; that holds for the cache itself but not for the
   VRAM a context actually consumes.
3. **The 4096 rung catches things.** First measured third-rung failure.
