# Model quantisation: which one should you download?

**English** · [繁體中文](model-quant.zh-TW.md)

A GGUF model comes in a dozen variants with names like `Q4_K_M`, `IQ4_XS`,
`Q5_K_S`. They are the same model at different precisions. This page is about
picking one when you have a single GPU and a fixed amount of VRAM.

## The one rule that matters most

**The largest quant that fits entirely in VRAM beats a better quant that
doesn't.**

If any part of the model spills into system RAM, every token has to cross the
PCIe bus. The slowdown is not subtle, and no quality gain from a higher precision
compensates for it.

Measured, same card, same model, same quant — only how much of it fits changes:

| | Generate |
|---|---|
| All 65 layers in VRAM | ~26 tok/s |
| 28 of 65 layers (rest in DDR4) | **5.11 tok/s** |

**Five times slower for spilling 57% of the layers.** Fitting is the primary
constraint; quality is what you optimise *within* what fits. Details, and when
spilling is nonetheless the right call, in [spill-cost.md](spill-cost.md).

## Reading the names

The number is the main thing: **bits per weight**.

| Name | Bits (approx) | Meaning |
|---|---|---|
| `Q8_0` | 8.0 | Near-lossless, twice the size of Q4 |
| `Q6_K` | 6.6 | Very good, rarely necessary |
| `Q5_K_M` | 5.7 | Good middle ground if it fits |
| `Q4_K_M` | 4.8 | The common default recommendation |
| `Q4_K_S` | 4.5 | Slightly smaller, slightly worse |
| `IQ4_XS` | 4.25 | Smallest Q4-class; needs an imatrix to make |
| `Q3_K_M` | 3.9 | Noticeable degradation begins |
| `IQ3_XXS` | 3.1 | Only when nothing else fits |

The letters:

- **`K`** — "K-quants", the standard modern scheme. Different tensors get
  different precision, because some matter more than others.
- **`S` / `M` / `L`** — small / medium / large within that family.
- **`I`** — "importance matrix" quants. Uses a calibration pass to decide which
  weights to protect, so it holds up better than its bit count suggests. `IQ4_XS`
  punches noticeably above 4.25 bits.

## The name does not fix the bits

The table above is approximate on purpose. Two files can carry the same quant
name and still differ by half a bit per weight, because "dynamic" builds — the
`UD-` prefix from Unsloth, and equivalents elsewhere — assign precision per
tensor rather than applying one scheme everywhere.

Bits per weight computed from ggml's actual block sizes, for two releases of
the same architecture:

| | Params | Tensor bytes | **bpw** |
|---|---|---|---|
| Qwen3.6-27B `IQ4_XS` | 26.90 B | 14,714 MiB | **4.5892** |
| Qwen3.8-27B `UD-IQ4_XS` | 27.32 B | 13,582 MiB | **4.1703** |
| Qwen3.8-27B `UD-Q4_K_S` | 27.32 B | 14,636 MiB | **4.4939** |

The middle row has *more* parameters in a *smaller* file than the top one, and
its name suggests they are the same thing. It is built from 12 distinct quant
types against the top row's 4, with 0.80 B parameters below 3 bits and
`token_embd` dropped from `Q4_K` to `Q3_K`.

**This is not a complaint about the packaging — it is a warning about
comparison.** On a 16 GB card that 0.42 bpw is worth 27,000 tokens of context,
so a newer model in a leaner file looks like a large generational improvement
and isn't. Matched at equal precision, the two land within 1.5% of each other:

| | Ceiling, q8_0 KV |
|---|---|
| Qwen3.6 `IQ4_XS` (4.5892 bpw) | 34,816 |
| Qwen3.8 `UD-IQ4_XS` (4.1703 bpw) | 61,952 |
| Qwen3.8 `UD-Q4_K_S` (4.4939 bpw) | 34,304 |

Full measurements: [rtx5060ti-16gb-qwen3.8-27b.md](../results/rtx5060ti-16gb-qwen3.8-27b.md).

To check a file before trusting its label, divide: **file bytes × 8 ÷ parameter
count**. Both numbers are on the Hugging Face model page, and llama.cpp prints
the parameter count when it loads. 14,252,845,984 × 8 ÷ 27.32 B = 4.17, which is
enough to tell a 4.2 bpw file from a 4.6 bpw one. The per-tensor breakdown needs
a GGUF parser, but you rarely need it — the average is what decides whether the
file fits.

## Working out what fits

Rough arithmetic, then verify:

```
file size on disk  +  KV cache for your context  +  ~1 GB overhead  <  usable VRAM
```

Two things people get wrong here, both of which cost real headroom:

- **Usable VRAM is not what `nvidia-smi` prints.** A 16 GB card reserves ~460 MiB
  for the driver, so the real ceiling is about 15.85 GB.
- **A desktop environment on the same card can hold 600+ MiB.** See
  [gotchas.md](gotchas.md).

A worked example from this project — Qwen3.6-27B on a 16 GB card:

| Quant | Size | Fits? |
|---|---|---|
| `Q4_K_M` | 16.82 GB | No — larger than the card before any KV cache |
| `Q4_K_S` | 15.86 GB | No — no room left for KV cache or overhead |
| **`IQ4_XS`** | **15.44 GB** | **Yes — 34,816 context with q8_0 KV** |
| `Q3_K_M` | 13.59 GB | Yes, with much more room to spare |

The usual advice is "use `Q4_K_M`". On this card that advice does not run at all.
`IQ4_XS` is the largest Q4-class quant that fits, so that is the right answer
here — and you only find that out by checking sizes against your actual card.

## Bigger model or better quant?

With a fixed VRAM budget you can have a larger model at lower precision, or a
smaller model at higher precision. The usual finding is that **the larger model
wins down to about 4 bits**, and below roughly 3 bits it stops being true.

So a 27B at `IQ4_XS` is generally a better bet than a 14B at `Q8_0` — but a 27B
at `IQ2` is usually worse than a 14B at `Q4_K_M`. Treat ~4 bits as the floor
worth targeting and pick the biggest model that reaches it.

## What about NVFP4, AWQ, GPTQ?

Different ecosystem. Those run on vLLM/TensorRT rather than llama.cpp, and
**vLLM cannot offload to CPU** — if the model doesn't fit, it doesn't start.

Also, names mislead. Both NVFP4 builds of Qwen3.6-27B are *larger* than the GGUF
Q4, because "NVFP4" is mixed precision: only the MLP tensors are 4-bit, while
attention stays at FP8 and (in one build) the vision tower stays at BF16.

| Build | Size |
|---|---|
| nvidia NVFP4 | 21.94 GB |
| unsloth NVFP4 | 23.44 GB |
| GGUF `IQ4_XS` | **15.44 GB** |

For a single consumer card, GGUF is usually the only thing that fits.

## Then check it actually runs

Sizes tell you what loads. They don't tell you what survives a full context
window — that's what this tool is for:

```bash
ctxprobe model.gguf
```
