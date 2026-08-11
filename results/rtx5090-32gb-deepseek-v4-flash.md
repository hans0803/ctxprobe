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
