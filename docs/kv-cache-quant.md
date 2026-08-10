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
`--cache-type-k q8_0 --cache-type-v q8_0` stores them at 8-bit instead: **half
the memory, for the same number of tokens.**

Measured on an RTX 5060 Ti 16GB with Qwen3.6-27B-IQ4_XS:

| KV type | Per 1K tokens | Max context on this card |
|---|---|---|
| `f16` (default) | ~68 MiB | see [results](../results/rtx5060ti-16gb-qwen3.6-27b.md) |
| `q8_0` | ~34 MiB | **34,816** |
| `q4_0` | ~17 MiB | untested here |

The freed memory converts directly into context. Nothing else about the model
changes — same weights, same speed.

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

**Start with `q8_0`.** It roughly doubles your context for a quality cost you
are unlikely to notice.

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
