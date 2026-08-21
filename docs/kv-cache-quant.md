# KV cache quantisation, and why you probably want q8_0

**English** · [繁體中文](kv-cache-quant.zh-TW.md)

If you have one consumer GPU and you want long context, this is the single
highest-leverage setting you have. This page explains what it does and what it
costs, with measured numbers rather than rules of thumb.

## What the KV cache is

When a model reads your prompt, each token produces two vectors — a **key** and
a **value** — at every attention layer. Those get kept around so that generating
token 5000 doesn't require re-reading tokens 1 through 4999 from scratch. That
store is the KV cache.

The important property: **it grows linearly with context length, and it lives in
VRAM alongside the model weights.**

```
VRAM = model weights (fixed)  +  KV cache (grows with context)  +  overhead
```

Weights are a one-time cost. The KV cache is the part that decides whether you
get 8K of context or 34K.

## What quantising it does

By default llama.cpp stores those vectors at 16-bit precision (`f16`). Setting
`--cache-type-k q8_0 --cache-type-v q8_0` stores them at 8-bit instead — which
sounds like half the memory for the same number of tokens, and isn't quite; see
[below](#it-saves-40-not-50).

Measured on an RTX 5060 Ti 16GB with Qwen3.6-27B-IQ4_XS:

| KV type | Bits per value | Max context on this card |
|---|---|---|
| `f16` (default) | 16 | 20,224 |
| **`q8_0`** | 8 | **34,816** |
| `q4_0` | 4 | untested here |

Both figures are measured, not estimated — see
[results](../results/rtx5060ti-16gb-qwen3.6-27b.md).

**That is 72% more context for one flag.** Nothing else about the model changes:
same weights, same output quality from the weights themselves.

Note it is *not* double, even though the cache is half the size. Weights are a
fixed cost the cache never touches — on this card 15.44 GB of the 15.85 GB usable
is model, leaving only ~400 MiB as cache budget. Halving the per-token cost
stretches that remainder, not the whole window. **The tighter the model fits, the
smaller the gain** — which is worth knowing before assuming q8_0 will rescue a
model that barely loads.

## It saves 40%, not 50%

`q8_0` holds 1.0625 bytes per value against `f16`'s 2, so halving is the natural
expectation. What a card actually gives up per token is measurably worse than
that.

Peak VRAM against context is linear enough to read straight off adjacent PASS
rows. Measured on Qwen3.8-27B, which has 16 full-attention layers with 4 KV
heads at `head_dim` 256 — 32,768 cache values per token:

| | Theory | Measured | Excess |
|---|---|---|---|
| `f16` | 64.0 MiB/1K | **63.48** | ~0 |
| `q8_0` | 34.0 MiB/1K | **38.09** | **+4.1** |

The f16 slope is the same figure on Qwen3.6, Qwen3.8 `UD-IQ4_XS` and Qwen3.8
`UD-Q4_K_S`, so it is a property of the architecture and it matches theory.
`q8_0` does not: it overshoots by 4,198 bytes per token, and one layer's
per-token KV held at f16 is 4 × 256 × 2 × 2 = 4,096 bytes — a 2.5% match.

The shape of that is a **dequantisation scratch buffer sized for one layer at a
time, growing linearly with context.** f16 needs no such buffer because nothing
has to be unpacked.

So the real ratio is 38.09 / 63.48 = **60% of the per-token cost, not 50%**. A
calculator that divides KV bytes by two will promise you more context than the
card delivers. Ceilings measured three ways:

| | f16 | q8_0 | Gain |
|---|---|---|---|
| Qwen3.6-27B `IQ4_XS` | 20,224 | 34,816 | +72.1% |
| Qwen3.8-27B `UD-IQ4_XS` | 36,352 | 61,952 | +70.4% |
| Qwen3.8-27B `UD-Q4_K_S` | 19,712 | 34,304 | +74.0% |

Still the highest-leverage flag available. Just not the factor of two.

Full run: [rtx5060ti-16gb-qwen3.8-27b.md](../results/rtx5060ti-16gb-qwen3.8-27b.md).

## What it costs

Quality loss from `q8_0` KV is small enough that it is hard to observe in normal
use, and it is the setting most long-context local setups run. 8 bits per value
is still a lot of precision for numbers that are being summed over.

`q4_0` KV is a different matter. Degradation becomes noticeable, and it shows up
in a specific way: the model gets vaguer about things far back in the context —
exactly the material you enabled long context for. Worth testing before relying
on it.

There is no meaningful speed cost either way. Generation speed on our card was
flat at ~26 tok/s regardless of KV type; the cache is a memory question, not a
throughput one.

## The practical recommendation

**Start with `q8_0`.** It buys a large amount of context for a quality cost you
are unlikely to notice.

This only matters once the weights themselves fit — see
[model-quant.md](model-quant.md) for that, and [spill-cost.md](spill-cost.md)
for what happens if they don't. No KV setting rescues a model that spills.

Reach for `f16` when you are doing something precision-sensitive and your
context is short enough that you can afford it. Reach for `q4_0` only when
`q8_0` still won't fit and you have tested that the output holds up.

```bash
llama-server -m model.gguf -ngl 999 -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1
```

## One thing that isn't obvious

KV cache cost depends on the model's architecture, not just its parameter count.
Qwen3.6-27B is 64 layers but only 16 of them use full attention — the other 48
are linear attention with a fixed-size state that does not grow with context. So
its KV cache is far cheaper than a conventional 27B model's would be.

This is why a number quoted for one model doesn't transfer to another of the
same size, and why measuring beats estimating. Related traps are in
[gotchas.md](gotchas.md) — particularly `--parallel`, which silently multiplies
your KV allocation by 4 and will undo everything on this page if you leave it at
the default.
