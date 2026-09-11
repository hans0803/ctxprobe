# RTX 5090 32GB — DeepSeek-V4-Flash-0731 (UD-IQ4_XS)

**English** · [繁體中文](rtx5090-32gb-deepseek-v4-flash.zh-TW.md)

Measured 2026-08-11. A 136.66 GB model running on one 32 GB card, with the
experts held in system RAM. This is the extreme end of the MoE offload case that
[Gemma4-26B-A4B](rtx5060ti-16gb-gemma4-26b-a4b.md) introduced: there, moving
experts out dropped the weights from 14.44 GB to 2.4 GB. Here it makes a 137 GB
model fit a consumer card at all.

## Hardware and model

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5090 — 32607 MiB reported, 32109 MiB allocatable** |
| Driver reserve | **498 MiB** (a 16 GB card reserves 462 — bigger cards reserve more) |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| RAM | **192 GB DDR5-5600**, dual channel |
| llama.cpp | `4801e3c` (2026-08-10) — `LLM_ARCH_DEEPSEEK4` landed after 6/26 |
| Model | [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) `UD-IQ4_XS`, **136.66 GB**, 4 shards |

The second 5090 in this box is serving something else; everything below uses a
single card via `CUDA_VISIBLE_DEVICES=1`.

Architecture, from the GGUF metadata:

```
deepseek4.block_count       = 43
deepseek4.expert_count      = 256
deepseek4.expert_used_count = 6      <- top-6 of 256
deepseek4.attention.key_length = 512  (MLA)
deepseek4.context_length    = 1048576
```

**Roughly 277 of ~280B parameters are experts.** That is what makes this work:
`--cpu-moe` evicts almost the entire model, and what stays on the GPU is only
attention, embeddings and shared experts.

## It runs, and context is not the constraint

| Context | Result | Peak VRAM | Prefill | Generate |
|---|---|---|---|---|
| 8,192 | PASS | 8927 MiB | 153.86 tok/s | 13.28 tok/s |
| **131,072** | **PASS** | 10441 MiB | 153.83 tok/s | 13.32 tok/s |

131,072 was **ctxprobe's search cap at the time, not the model's limit** — the
binary search ended after two boots because the upper bound itself passed. The
model declares 1,048,576. The cap has since been raised to 262,144; this row is
a lower bound until it is re-measured.

Those two rows cleared the prompt ladder (64/512/4096) but not, at the time,
the 95% fill validation: at 153 tok/s the validating prompt for 131,072 needs
124K tokens of prefill — 13 minutes — so it was stopped, and 131,072 stood as
*ladder-verified, unconfirmed*.

**Confirmed 2026-08-31**, with the ubatch finding below making it practical
(775 tok/s instead of 153) and a ctxprobe that bisects when a full window
fails:

```
Validating 131072 at 95% fill...
  confirmed: 131072 (124075 tokens filled)
  peak VRAM 15469 MiB | prefill 653.41 tok/s | generate 11.92 tok/s
```

Two and a half minutes, at `-ub 8192`. The ladder was right here. It is worth
saying why, because on [Qwen3.8-Flash-Next
](rtx5090-32gb-qwen3.8-flash-next.md) — also sparse attention, also an indexer
whose scratch grows with `n_kv` — the ladder overstated a ceiling by 41%. The
difference is not the mechanism, it is the slack: this configuration peaks at
15,469 MiB of 32,109, so whatever the indexer's working set does with a 124K
window, there are 16 GB for it to do it in. Qwen3.8-Flash-Next was measured
with 10 MiB spare. **A growing buffer only becomes a ceiling on a full card.**

The validated speeds are lower than the table's, as they should be: 653 tok/s
prefill and 11.92 tok/s generate against a 124K-token window, versus 154 and
13.32 measured at `-ub 512` against a short one. Peak VRAM is 15,469 rather
than 10,441 for the same reason the ubatch table shows — the compute buffer is
sized by `-ub`, and this ran at 8192.

KV cost works out at about **12.3 KB per token** (from the 1,514 MiB gap between
the two rows). MLA is extraordinarily cheap here: Qwen3.6-27B costs 34 MiB per
1K tokens, this costs 12.3 — a factor of 2,800.

Base VRAM is **~8.8 GB** before KV. That is well above what the parameter counts
alone predict (~2.3 GB); this llama.cpp build does not print buffer allocations,
so the breakdown is unresolved. Reported as measured rather than explained.

## What it costs in system RAM

The card is the cheap half of this deployment. The expert set has to live
somewhere, and that somewhere is RAM.

Same config (`--cpu-moe -ub 8192 -c 24576 --threads 28`), the only difference
being how llama.cpp gets the file into memory:

| | mmap (default) | `--no-mmap` |
|---|---|---|
| Load time | **4 s** | **75 s** |
| Process RSS | 131,061 MiB | 124,730 MiB |
| System `used` growth | **+1.8 GB** | **+126 GB** |
| `available` after load | 177 GB | 53 GB |
| Generate, 3 × 256 tokens | 12.03 / 12.08 / 12.07 | 12.07 / 12.08 / 12.12 |

**RSS lies under mmap.** The process reports 131 GB resident while the system's
`used` grows by 1.8 GB, because those pages are page cache backing a file
mapping — shared, clean, and evictable. Reading RSS as "this process needs
131 GB of RAM" is wrong in one direction; reading the 1.8 GB as the requirement
is wrong in the other.

**`--no-mmap` buys nothing here.** llama.cpp prints
`consider using --no-mmap for better performance` whenever CPU tensor overrides
are combined with mmap, so it looks like a free win. Measured, generation is
identical to within 0.4% — the same figure three times over — while the cost is
126 GB of *anonymous* memory that the kernel cannot reclaim and a load that takes
19× longer. Once the mapping is fully resident, there is nothing left for
`--no-mmap` to fix.

The `--no-mmap` RSS being 5.6 GB *smaller* than the 130,332 MiB file is a useful
consistency check: that is the non-expert tensors, which went to the GPU and were
never allocated in RAM.

### How much do you actually need

Routing makes this simple. Each token picks 6 of 256 experts per layer, so
consecutive tokens touch different experts, and across a few dozen tokens you
have touched essentially all of them. There is no working set smaller than the
model.

> **System RAM ≥ model file size + ~10 GB.**

For this model that is 130,332 MiB of weights, so:

| RAM | Verdict |
|---|---|
| 96 GB | No. Every token faults from disk; expect single-digit *seconds* per token |
| 128 GB | Borderline — the file alone is 127.3 GiB, leaving nothing for the OS |
| **192 GB** | **Comfortable.** 53 GB spare here even in the `--no-mmap` worst case |

The failure mode when RAM is short is not an error. It loads, it answers, and it
is catastrophically slow, because the page cache thrashes against storage instead
of DDR5 — the same shape as the spill problem one level down the hierarchy.

Keep `buff/cache` in view: on this box it sits at ~160 GB with the model warm,
and that number *is* the model being resident.

## Speed is bandwidth-bound, and the arithmetic says so

Generation at ~13.3 tok/s with 137 GB of experts in DDR5 looks surprising until
the routing is counted:

```
each expert   3 × 4096 × 2048 = 25.17M parameters
per token     43 layers × (6 routed + 1 shared) = 301 experts
              301 × 25.17M × ~0.49 bytes ≈ 3.7 GB per token

13.3 tok/s × 3.7 GB = 49 GB/s
```

Measured on this host: 37.0 GB/s copy, 35.4 GB/s single-threaded read. 49 GB/s
across many threads is consistent with a saturated dual-channel DDR5-5600 bus.
**It is bandwidth bound, exactly as a 4B-active model over DDR5 should be.**

That 49 GB/s is a slight underestimate, because it divides by the whole token.
[The scaling law](#the-scaling-law) below separates out 16.0 ms of fixed cost
that is not memory traffic, leaving 3.7 GB over 66.8 ms — **55 GB/s** for the
expert read itself.

## The ubatch finding

Prefill started at 153 tok/s — 23× slower than Gemma4 on a much weaker card. The
reason is that prefill defeats sparsity: in a 512-token ubatch, each token picks
6 of 256 experts, so the batch collectively touches nearly all of them. **Every
ubatch reads essentially the whole expert set for a layer**, and that cost is
amortised over however many tokens share the batch.

Sweeping `-ub` (and `-b` with it), 20,480-token prompt, `-c 24576`:

| ubatch | Peak VRAM | Prefill | vs 512 | Generate |
|---|---|---|---|---|
| 512 (default) | 9057 MiB | 151.51 | — | 12.38 |
| 1,024 | 9291 MiB | 259.66 | 1.7× | 12.32 |
| 2,048 | 9767 MiB | 426.83 | 2.8× | 12.26 |
| 4,096 | 10649 MiB | 649.81 | 4.3× | 12.12 |
| **8,192** | **12509 MiB** | **775.35** | **5.1×** | 12.07 |
| 16,384 | 17503 MiB | 782.03 | 5.2× | 11.75 |
| 24,576 | 23003 MiB | 767.27 | 5.1× | 11.67 |

**8,192 is the knee.** Past it prefill is flat within noise while VRAM climbs
from 12.5 GB to 23 GB — the compute buffer is sized by ubatch whether or not the
prompt fills it. Generation is unaffected throughout (12.38 → 11.67), which is
the control that makes the prefill result unambiguous: batch-of-1 decoding does
not care about ubatch.

The knee is not a constant. 8,192 splits this 20K prompt into 3 passes; by then
the expert read is essentially amortised. Expect the useful value to track
"large enough to split your prompt into 2-3 passes".

### Where the payoff starts

Same model, `-ub 512` against `-ub 8192`, measuring wall-clock time to first
token:

| Prompt | `-ub 512` | `-ub 8192` | Speedup |
|---|---|---|---|
| 140 | 2,974 ms | 2,998 ms | 1.00× |
| 249 | 3,370 ms | 3,400 ms | 0.99× |
| 511 | 3,894 ms | 3,929 ms | 0.99× |
| 998 | 6,857 ms | 4,312 ms | **1.59×** |
| 2,020 | 13,390 ms | 5,133 ms | **2.61×** |

**Nothing below 512, everything above it.** The break lands exactly on
`n_ubatch`, which is the mechanism showing itself rather than being argued for.

Two things fall out of the `-ub 512` column:

- Each additional ubatch costs **~3 seconds**, regardless of how many tokens are
  in it. That is one pass over a layer's experts through DDR5.
- Even a 140-token prompt waits ~3 seconds. **TTFT has a floor of about 3
  seconds** in this deployment shape — the first expert sweep is unavoidable.

So `-ub 8192` costs 3.5 GB of VRAM and is worth nothing for short chat turns,
1.6-2.6× for prompts with a system prompt and some history, and 5× for documents.
On a 32 GB card with ~20 GB spare, there is no reason not to set it.

## Keeping some experts on the GPU

`--cpu-moe` evicts every expert. `--n-cpu-moe N` keeps the experts of the first
N layers in RAM, so a smaller N leaves more on the card. Single GPU, `-ub 8192`,
`-c 24576`:

| `--n-cpu-moe` | GPU layers | Peak VRAM | Prefill | Generate, 3 × 256 tok |
|---|---|---|---|---|
| 43 (= `--cpu-moe`) | 0 | 12,243 MiB | 621.11 | 12.09 / 12.04 / 12.07 |
| 41 | 2 | 17,939 MiB | — | 12.53 / 12.57 / 12.57 |
| 40 | 3 | 20,787 MiB | 643.99 | 12.73 / 12.85 / 12.86 |
| 39 | 4 | 23,635 MiB | — | 12.92 / 13.07 / 13.08 |
| **38** | **5** | **26,481 MiB** | 669.30 | **13.33 / 13.35 / 13.35** |
| 36 | 7 | ✗ OOM | — | — |

The VRAM steps are exactly linear — 2,848 MiB per additional layer. That figure
predicted every capacity result below, so it is worth deriving for any model you
plan to place by hand.

Five layers is the ceiling on 32 GB, and it buys **+11% generate for 14 GB**.

### The scaling law

Those five points fit one line. Taking time per token against the fraction of
expert layers still in RAM:

```
t_token  =  16.0 ms  +  66.8 ms × (43 − K) / 43        K = layers on the GPU
```

Every measured point lands within 0.2 ms of it:

| K | Measured | Predicted |
|---|---|---|
| 0 | 82.85 ms | 82.8 |
| 2 | 79.62 ms | 79.7 |
| 3 | 78.06 ms | 78.1 |
| 4 | 76.80 ms | 76.6 |
| 5 | 74.96 ms | 75.0 |

The two terms are the whole story of this deployment shape:

- **16.0 ms is fixed.** Attention, the shared expert, the router, kernel launch
  and synchronisation. No amount of VRAM removes it, and it sets the ceiling:
  with every expert on the card, this model tops out around **62 tok/s**.
- **66.8 ms is the DDR5 expert read**, and it is the only part VRAM can buy
  back — strictly in proportion to the *fraction* of layers moved.

That proportionality is what makes VRAM a poor purchase here. Each layer costs a
fixed 2,848 MiB and returns a fixed 1.55 ms, so the return on a gigabyte is
roughly constant while the thing being improved is already small:

| Target | Layers needed | Expert VRAM | Total VRAM | Hardware |
|---|---|---|---|---|
| 12 tok/s (baseline) | 0 | 0 | **12.3 GB** | one 16 GB card |
| 13.3 tok/s (+11%) | 5 | 13.9 GB | 26.5 GB | one 32 GB card |
| 18 tok/s (+50%) | ~18 | 50 GB | ~60 GB | two 32 GB cards |
| 24 tok/s (+100%) | ~27 | 76 GB | **~85 GB** | three 32 GB cards |
| 62 tok/s (ceiling) | 43 | 120 GB | ~128 GB | four 32 GB cards |

Rows past 5 layers are extrapolation from the fit, not measurement, and the
two-card section below is the one place it was tested — where it did not hold.

### Both cards

Two 5090s, experts placed explicitly across them. Layer ranges split evenly, one
regex per `-ot` flag:

| Layers on GPU | GPU0 + GPU1 | Total | Prefill | Generate |
|---|---|---|---|---|
| 7 | 19,775 + 16,873 | 36.6 GB | 718.85 | 13.84 |
| 13 | 25,471 + 28,263 | 53.7 GB | 821.51 | 15.86 |
| 14 | 28,319 + 28,263 | 56.6 GB | 841.92 | 13.70 |
| 15 | 31,167 + 28,263 | 59.4 GB | 860.60 | 14.34 |
| 16 | 31,167 + 31,111 | **62.3 GB** | **884.79** | 13.38 |
| 17 | ✗ OOM | 65.1 GB predicted | — | — |

Capacity behaves exactly as 2,848 MiB/layer predicts: 16 layers fit at 62.3 GB
against 64.2 GB allocatable, 17 does not.

Placement is verifiable from VRAM alone, which is worth doing given how quietly
`-ot` fails. Every row matches 2,848 MiB per layer on top of the two baselines
(8,381 and 8,329 MiB) to within 3 MiB:

| Asked for | GPU0 | Predicted | GPU1 | Predicted |
|---|---|---|---|---|
| 4 + 3 layers | 19,775 | 8,381 + 4×2,848 = 19,773 | 16,873 | 8,329 + 3×2,848 = 16,873 |
| 6 + 7 layers | 25,471 | 25,469 | 28,263 | 28,265 |
| 8 + 8 layers | 31,167 | 31,165 | 31,111 | 31,113 |

2,848 MiB is **all 256 experts of one layer** — 43 × 2.98 GB accounts for the
128 GB file. So each layer is either wholly on a card or wholly in RAM, and
routing cannot pull an expert off-card. That rules out the obvious reading of
the generate column.

**Prefill is monotonic** (719 → 885, +23% over the range) because it is compute
bound once ubatch has fixed the bandwidth problem, and a second card adds
compute.

**The generate column is unreliable.** These rows sampled 32 generated tokens,
which is not enough to time — the single-card ladder was re-run at 256 tokens
and tightened from that scatter to ±0.2%. Read the two-card generate figures as
13–16 with no resolvable ordering, and note that the scaling law predicts **17.3
tok/s at 16 layers**, roughly 25% above anything measured here. Whether the
second card genuinely fails to deliver the law or the sample was simply too
short is **unresolved**; it needs a re-run at 256 tokens.

This writeup previously explained the scatter as per-token routing moving
experts on and off the card. The placement arithmetic above rules that out. It
is recorded in [gotchas #10](../docs/gotchas.md) as the mistake it was.

### Is the second card worth it

| Setup | Generate | Prefill | VRAM |
|---|---|---|---|
| Single card, `--cpu-moe` | **12.07** ± 0.2% | 621 | **12.3 GB** |
| Single card, `--n-cpu-moe 38` | **13.34** ± 0.1% | 669 | 26.5 GB |
| Two cards, 13–16 layers | 13–16, unresolved | 821 – 885 | 53.7 – 62.3 GB |

Prefill is the honest case for the second card: **+23%, measured over thousands
of tokens**, which is a figure that holds. Set against `-ub 8192` buying 5.1×
prefill for 3.5 GB, the exchange rate is still poor.

For generation, the second card cannot be shown to have bought anything at all
above the single-card 13.34 — and the law says the most it could buy is +30%.

## What does not work

**`--tensor-split` does nothing here.** Both `1,1` and `3,1` failed to load at
13 layers, and the first dual-GPU attempt without `-ot` left GPU0 at 8.4 GB while
GPU1 hit 31.1 GB.

The reason is that two mechanisms compose badly. Layer split is *contiguous*:
GPU0 takes an early block of layers, GPU1 the rest. `--n-cpu-moe N` leaves the
GPU-side experts in the *last* N layers — one contiguous block, which lands
entirely on one card. Moving the split point cannot separate them; only naming
tensors explicitly can.

Two ways to get `-ot` silently wrong, both of which produced a table full of
plausible numbers before being caught:

```bash
# WRONG — the CPU pattern comes first and swallows everything
-ot "blk\.(3[0-9])\.ffn_.*_exps\.weight=CPU" -ot "blk\.30\..*=CUDA0"

# WRONG — greedy .* eats the comma separator, matching nothing
-ot "blk\.36\.ffn_.*_exps\.weight=CUDA0,blk\.37\.ffn_.*_exps\.weight=CUDA0"

# RIGHT — one regex range per flag, -ot before --cpu-moe
-ot "blk\.(3[0-5])\.ffn_.*_exps\.weight=CUDA0" \
-ot "blk\.(3[6-9]|4[0-2])\.ffn_.*_exps\.weight=CUDA1" \
--cpu-moe
```

Neither mistake reports an error. The tell is VRAM sitting at the
no-experts-on-GPU baseline — 8,381 + 8,329 MiB here — while generate stays at
the all-CPU figure.

## What this says about buying hardware

Once a model cannot fit in VRAM, VRAM stops being the thing worth buying.

| VRAM | Config | Generate | Card it needs |
|---|---|---|---|
| **9.1 GB** | `--cpu-moe`, `-ub 512` | 12.4 | 12 GB |
| **12.3 GB** | `--cpu-moe`, `-ub 8192` | 12.07 | **16 GB** |
| 26.5 GB | `--n-cpu-moe 38` | 13.34 | 32 GB |
| 62.3 GB | two cards, 16 layers | 13–16 | two 32 GB |

**A 137 GB model runs at 12 tok/s on a 16 GB card.** Five times that VRAM is
worth a confirmed +11%. The reason is in the scaling law: you buy expert layers
linearly at 2,848 MiB each, and each one returns a fixed 1.55 ms against a
74–83 ms token. Nothing interesting happens until most of the model is resident,
and "most of a 137 GB model" is not a consumer purchase.

16 GB rather than 12 GB is the one place the extra VRAM clearly pays, and it is
not for the weights: it buys `-ub 8192`, which is worth **4.1× prefill for
3.5 GB**. Everything above that is buying the flat part of a hyperbola.

So the shape of the recommendation inverts:

> Buy the smallest card that holds attention, KV and an 8192 compute buffer.
> Spend the rest on RAM — capacity first so the model is resident at all, then
> bandwidth, which is what actually sets the speed.

Bandwidth is the untested lever, and the law says it is the large one. The
66.8 ms expert read is 3.7 GB/token across a saturated dual-channel bus. On an
8-channel platform at ~200 GB/s that term falls to ~17 ms, putting the token at
~33 ms — **~30 tok/s, from memory controllers rather than VRAM**. That is a
projection from the fit, not a measurement, and the 16.0 ms fixed term may not
survive the move; but it is 2.5× against the ~85 GB of VRAM that 2× would cost.

### This does not generalise past MoE

The whole argument rests on 6 of 256 experts being read per token. A dense model
reads *everything* per token, so the same "let system RAM hold the overflow"
move collapses. Measured on the same simulated 8 GB budget in
[spill-cost.md](../docs/spill-cost.md):

| | In system RAM | VRAM used | Generate |
|---|---|---|---|
| Gemma4-26B-A4B (MoE) | the experts, via `--cpu-moe` | 2.4 GB | **39.7 tok/s** |
| Qwen3.6-27B (dense) | 37 of 65 layers, via `-ngl 28` | ~8 GB | **5.1 tok/s** |

Same card, same failure to fit, an 8× difference in what it costs. For a sparse
MoE, not fitting is a manageable trade and the card should be sized down. For a
dense model, not fitting is the problem, and the answer is a smaller quant or a
smaller model — never more system RAM.

## Practical configuration

```bash
llama-server -m DeepSeek-V4-Flash-0731-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 999 -c 131072 --parallel 1 --cpu-moe \
  -b 8192 -ub 8192 --threads 28
```

`--threads 28` leaves 2 physical cores for other services on the box; it costs
about 7% of generation speed (13.3 → 12.4 tok/s), since CPU threads drive the
DDR5 reads.

`--cpu-moe` is not optional here — without it the model cannot load at all.
