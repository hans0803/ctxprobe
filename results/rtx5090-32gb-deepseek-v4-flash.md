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

131,072 is **ctxprobe's search cap, not the model's limit** — the binary search
ended after two boots because the upper bound itself passed. The model declares
1,048,576.

⚠️ **Both figures cleared the prompt ladder (64/512/4096) but not the 95% fill
validation.** At 153 tok/s the validating prompt for 131,072 needs 124K tokens of
prefill — 13 minutes — so it was stopped. Treat 131,072 as *ladder-verified,
unconfirmed*. The ubatch findings below make that validation practical again.

KV cost works out at about **12.3 KB per token** (from the 1,514 MiB gap between
the two rows). MLA is extraordinarily cheap here: Qwen3.6-27B costs 34 MiB per
1K tokens, this costs 12.3 — a factor of 2,800.

Loading takes **8 seconds**, not minutes, because llama.cpp mmaps the file and
faults pages in on demand. `buff/cache` sits at ~160 GB afterwards, so the
experts live in page cache rather than as a second copy in RAM.

Base VRAM is **~8.8 GB** before KV. That is well above what the parameter counts
alone predict (~2.3 GB); this llama.cpp build does not print buffer allocations,
so the breakdown is unresolved. Reported as measured rather than explained.

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

| `--n-cpu-moe` | GPU layers | Peak VRAM | Prefill | Generate |
|---|---|---|---|---|
| 43 (= `--cpu-moe`) | 0 | 12,345 MiB | 621.11 | 12.08 |
| 40 | 3 | 20,889 MiB | 643.99 | 12.91 |
| **38** | **5** | 26,585 MiB | 669.30 | **13.37** |
| 36 | 7 | ✗ OOM | — | — |

The VRAM steps are exactly linear — 8,544 MiB for 3 layers, 5,696 for 2 — giving
**2,848 MiB per layer**. That figure predicted every capacity result below, so it
is worth deriving for any model you plan to place by hand.

Five layers is the ceiling on 32 GB, and it buys **+11% generate for 14 GB**.

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

**Prefill is monotonic** (719 → 885, +23% over the range) because it is compute
bound once ubatch has fixed the bandwidth problem, and a second card adds
compute.

**Generate is not monotonic**, and that is not measurement noise. Routing is
per-token: each token picks 6 of 256 experts, and `-ot` places experts *by
layer*, not by how often they are used. Whether a given token's experts happen
to sit on a card or in DDR5 varies with what the model generates, so throughput
varies with the content. A fixed tok/s figure does not exist for this
configuration — only a range, here roughly **13.4 to 15.9**.

### Is the second card worth it

| Setup | Generate | Prefill | VRAM |
|---|---|---|---|
| Single card, `--n-cpu-moe 38` | 13.37 | 669 | 26.6 GB |
| Two cards, 13-16 layers | 13.4 – 15.9 | 821 – 885 | 53.7 – 62.3 GB |

Roughly **+18% generate and +23% prefill for an entire second 32 GB card**. Set
against `-ub 8192`, which bought 5.1× prefill for 3.5 GB, the exchange rate is
poor. Worth it only if prefill specifically matters to you.

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
