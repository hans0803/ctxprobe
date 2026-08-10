# What it costs when the model doesn't fit

**English** · [繁體中文](spill-cost.zh-TW.md)

Every other page here assumes the goal is keeping the whole model in VRAM. This
one is about what happens when you don't, and why that assumption is worth
having in the first place.

## The short version

Same card, same model, same quant. The only thing that changes is how much of it
fits in VRAM:

| | Generate |
|---|---|
| All 65 layers in VRAM | ~26 tok/s |
| 28 of 65 layers, rest in system RAM | **5.11 tok/s** |

**Five times slower for spilling 57% of the layers.**

llama.cpp will happily do this for you. Pass `-ngl` lower than the layer count,
or just let it fall back, and it runs — it simply runs at a fraction of the
speed, with no error to tell you why.

## Why it's this bad

A layer living in system RAM has to be reached across the PCIe bus for every
single token. VRAM on this card runs at 448 GB/s; PCIe 5.0 x8 tops out around
32 GB/s in theory and less in practice. That gap is the whole story.

It also cannot be hidden by a faster CPU or more RAM. The bottleneck is the
link, not the memory or the compute at the far end of it. Faster DDR won't fix
it; more channels won't fix it.

This is why "the largest quant that *fits* beats a better one that doesn't" is
the first rule in [model-quant.md](model-quant.md). A one-step-better quant
might buy you a few percent of quality. Spilling costs you 80% of your speed.

## Measuring it on hardware you don't have

The numbers above come from an RTX 5060 Ti 16GB pretending to be the 8GB
version. That substitution is legitimate here for a specific reason: the two
variants are the **same GB206 die, same 4608 CUDA cores, same clocks, and the
same 128-bit GDDR7 at 448 GB/s**. Capacity is the only difference on the spec
sheet, so constraining the larger card reproduces the smaller one closely.

**Check that before borrowing this trick.** Most cards sold in two capacities cut
the memory bus along with the chips — an RTX 3060 12GB and 8GB differ in bus
width, so a constrained 12GB card would not tell you what the 8GB one does.
Verify same die, same core count, same bandwidth first.

The method is to hold VRAM with a separate CUDA process
([tools/hold-vram.py](../tools/hold-vram.py)) so the run under test sees a
smaller budget:

```bash
# leave 7800 MiB free — roughly a real 8GB card's allocatable budget
python tools/hold-vram.py 7800 &
ctxprobe model.gguf --ngl 28 --list 4096
kill %1
```

Two caveats worth stating in any report produced this way:

- **The driver reserve differs.** A 16GB card here reserves 462 MiB; an 8GB card
  reserves less. Targeting 7800 MiB free is an estimate, and a conservative one.
- **`nvidia-smi` figures include the holding process.** Subtract it to get the
  model's own usage. `ctxprobe` reports free VRAM separately for this reason.

## What the 8GB case actually showed

| `-ngl` | Layers on GPU | Result | Prefill | Generate |
|---|---|---|---|---|
| 999 | all 65 | **LOAD_FAIL** | — | — |
| 32 | 32 | **LOAD_FAIL** | — | — |
| 28 | 28 of 65 | PASS | 391 tok/s | 5.11 tok/s |
| 24 | 24 of 65 | PASS | 368 tok/s | 4.63 tok/s |

A 27B at `IQ4_XS` (15.44 GB) does not fit an 8GB card in any usable sense.

The instinct at that point is to reach for a smaller quant — and that instinct
is wrong here. `Q3_K_M` is 13.59 GB. `IQ3_XXS` is 11.99 GB. Neither fits either,
and both are meaningfully worse models. **The answer to "27B won't fit my 8GB
card" is a smaller model, not a smaller quant.** A 14B at `Q4_K_M` would fit
with room for context and run at full speed.

## When spilling is actually fine

It isn't always the wrong call:

- **You need the bigger model's capability more than speed**, and 5 tok/s is
  tolerable for your use. Reading speed is roughly 5-10 tok/s, so this is not
  automatically unusable — it's unusable for anything agentic or long-form.
- **MoE models.** Only a fraction of the parameters are active per token, so
  keeping experts in system RAM (`--n-cpu-moe`) costs far less than spilling
  dense layers. That is a genuinely different trade, not the one measured here.

What isn't fine is spilling *by accident* — which is the common case, because
nothing announces it. Check the layer count in the startup log, or run
`ctxprobe`, which refuses to call a partial offload a pass.
