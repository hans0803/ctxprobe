# Model quantisation: which one should you download?

**English** · [繁體中文](model-quant.zh-TW.md)

A GGUF model comes in a dozen variants with names like `Q4_K_M`, `IQ4_XS`,
`Q5_K_S`. They are the same model at different precisions. This page is about
picking one when you have a single GPU and a fixed amount of VRAM.

## The one rule that matters most

**The largest quant that fits entirely in VRAM beats a better quant that
doesn't.**

If any part of the model spills into system RAM, every token has to cross the
PCIe bus. The slowdown is not subtle — it is commonly 5-10× — and no quality
gain from a higher precision compensates for it. Fitting is the primary
constraint; quality is what you optimise *within* what fits.

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
