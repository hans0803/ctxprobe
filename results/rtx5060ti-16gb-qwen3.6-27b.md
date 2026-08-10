# RTX 5060 Ti 16GB — Qwen3.6-27B-IQ4_XS

**English** · [繁體中文](rtx5060ti-16gb-qwen3.6-27b.zh-TW.md)

Full measurement run, 2026-08-10. This is the data the tool was built from.
Covers both `q8_0` and `f16` KV cache, so the cost of not quantising it is
visible rather than assumed.

Reproduce the ceiling in six boots:

```bash
ctxprobe Qwen3.6-27B-IQ4_XS.gguf --min 32768 --max 36864 -- --reasoning-budget 0
```

## Hardware

Everything here is decided by the card. The model runs entirely in VRAM
(`-ngl 999`, no CPU offload), so host RAM never enters the budget.

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 16311 MiB reported, 15849 MiB allocatable** |
| Driver | 595.84 |
| llama.cpp | `0a50d99`, CUDA build |
| Host | i7-14700, 64 GB DDR4 — not a factor in any number below |

## Model

[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — `Qwen3.6-27B-IQ4_XS.gguf`, 15.44 GB.

Qwen3.6-27B is 64 layers but only **16 are full attention** (`full_attention_interval: 4`);
the other 48 are linear attention with fixed-size state. KV cache is therefore much
cheaper than a conventional 27B — about **34 MiB per 1K tokens** at q8_0.

Why IQ4_XS: it is the largest Q4-class quant that fits. `Q4_K_S` is 15.86 GB and
`Q4_K_M` is 16.82 GB — both exceed the card before any KV cache exists. How to
make that choice on your own card: [model-quant.md](../docs/model-quant.md).

## Context ceiling

`--parallel 1`, `--cache-type-k/v q8_0`, `-ngl 999`, `--reasoning-budget 0`.

| Context | Result | Peak VRAM | Generate |
|---|---|---|---|
| 8,192 | PASS | 15420 MiB | 25.88 tok/s |
| 16,384 | PASS | 15722 MiB | 25.90 tok/s |
| 20,480 | PASS | 15435 MiB | 25.99 tok/s |
| 24,576 | PASS | 15591 MiB | 25.97 tok/s |
| 28,672 | PASS | 15747 MiB | 26.01 tok/s |
| 30,720 | PASS | 15825 MiB | 26.00 tok/s |
| 33,792 | PASS | 15797 MiB | 26.01 tok/s |
| **34,816** | **PASS** | 15845 MiB | **25.99 tok/s** |
| 35,072 | **LONG_OOM** | 15847 MiB | 25.98 tok/s |
| 35,328 | FAIL | — | — |
| 36,864 | FAIL | — | — |

No prefill column here on purpose. These runs were qualified with a short
prompt, and a short prompt reports ~98 tok/s prefill on this card versus ~900
tok/s under a real one — an artefact of fixed overhead, not a rate. Prefill is
only meaningful measured against prompt length, which is the next section.

**34,816 is the ceiling**, re-confirmed with the window filled to 95%
(32,997 tokens measured through the server's tokenizer). 35,072 loads, decodes a
short prompt at full speed, then CUDA-OOMs on a real prompt and leaves a defunct
process behind. It is the single clearest argument for validating with a
full-size prompt.

Generation speed is flat at ~26 tok/s across the whole range — context costs
memory, not throughput, until you hit the wall.

## KV cache: q8_0 vs f16

Same model, same card, only `--cache-type-k/v` changed. Both ceilings found by
binary search with the window filled to 95%. What the setting does and when not
to use it: [kv-cache-quant.md](../docs/kv-cache-quant.md).

| KV type | Max context | Peak VRAM | Prefill | Generate |
|---|---|---|---|---|
| `f16` (default) | 20,224 | 15843 MiB | 922.54 tok/s | 26.94 tok/s |
| **`q8_0`** | **34,816** | 15845 MiB | 858.39 tok/s | 24.62 tok/s |

**Quantising the KV cache bought 72% more context** (14,592 extra tokens) on this
setup.

The obvious guess is that halving the cache should double the context, and it
doesn't. Weights are a fixed 15.44 GB that the cache never touches — only the
leftover ~400 MiB is cache budget, so halving the per-token cost extends that
leftover rather than the whole window. The bigger the model relative to the card,
the smaller the multiplier.

The f16 search in full:

| Context | Result | Peak VRAM | Filled |
|---|---|---|---|
| 8,192 | PASS | 15081 MiB | 7,733 |
| 16,384 | PASS | 15601 MiB | 15,424 |
| 18,432 | PASS | 15731 MiB | 17,287 |
| 19,456 | PASS | 15795 MiB | 18,324 |
| 19,968 | PASS | 15827 MiB | 18,897 |
| **20,224** | **PASS** | 15843 MiB | 19,203 |
| 20,480 | DECODE_OOM | 15783 MiB | — |
| 24,576 | LOAD_FAIL | — | — |

Don't read the generation column as "f16 is faster". Those runs sit at 20K
context against q8_0's 34K, and generation slows as the window fills — at a
comparable 16,384 the f16 run gave 27.28 tok/s, within noise of the q8_0 figures
at similar depth. KV type is a memory decision, not a speed one.

## Lazy vs eager kernel loading

CUDA 12 loads a kernel's code the first time something touches it. That makes
the ceiling depend on which kernels a given run happened to reach.
`CUDA_MODULE_LOADING=EAGER` loads all of them at startup instead:

| Module loading | Max context | Difference |
|---|---|---|
| `LAZY` (default) | 34,816 | — |
| **`EAGER`** | **25,344** | **−9,472 (−27%)** |

That gap is not a rounding error. It is how much of the 34,816 was resting on
"this workload never instantiates another kernel". Change the sampler, add a
grammar, feed a different batch shape, and a config that passed can still die in
production — the LAZY number is a property of the test, the EAGER number is a
property of the config.

The two also fail in different places, which confirms the mechanism:

| Mode | Where it fails |
|---|---|
| LAZY | `cudaFuncSetAttribute` at `mmq.cuh:1375` — loading a kernel |
| EAGER | `alloc` at `ggml-cuda.cu:589` — the CUDA memory pool |

Note EAGER does **not** make the prompt ladder redundant. 25,600 still passes a
short prompt and dies on a longer one, because ggml's pool grows on demand
regardless of when kernels were loaded. A single successful boot is not a pass
in either mode.

Which number to use: `LAZY` if you control the workload and want the most
context, `EAGER` if the config has to survive whatever gets thrown at it.

## Prefill vs prompt length

Measured against the deployed 34,816 config. Each prompt is randomly generated
so the server's prompt cache can't skew results.

| Prompt tokens | Prefill | TTFT | Generate |
|---|---|---|---|
| 799 | 836 tok/s | 0.96 s | 27.52 tok/s |
| 3,230 | 970 tok/s | 3.3 s | 26.67 tok/s |
| 12,649 | 945 tok/s | 13.4 s | 25.56 tok/s |
| 21,577 | 905 tok/s | 23.8 s | 24.94 tok/s |
| 25,424 | 890 tok/s | 28.6 s | 23.99 tok/s |
| 32,508 | 860 tok/s | 37.8 s | 23.50 tok/s |

Prefill holds ~860–970 tok/s with only mild decay. TTFT grows linearly and is
the dominant cost at long context — 38 seconds before the first token at 32K.
Generation drops ~13% from empty to full window.

Short prompts are *slower* per token (836 tok/s at 799 tokens) because fixed
overhead hasn't amortised.

## What freeing VRAM bought

The context ceiling moved twice without touching the model:

| Change | dGPU idle usage | Ceiling |
|---|---|---|
| Default (`--parallel 4`), desktop on dGPU | 605 MiB | 4,096 |
| `--parallel 1` | 605 MiB | 16,384 |
| Display moved to iGPU | 162 MiB | 30,720 |
| GNOME Remote Desktop moved to iGPU | **15 MiB** | **34,816** |

`--parallel 1` alone was a 4× win. Moving the display to the integrated GPU
(BIOS: `Primary Display = IGFX`, `iGPU Multi-Monitor = Enabled`) freed 443 MiB.

GNOME Remote Desktop kept holding 130 MiB on the dGPU even after the display
moved, because it loads `libcuda` + `libnvidia-encode` for NVENC. Pinning it to
Mesa/Intel released it:

```ini
# ~/.config/systemd/user/gnome-remote-desktop.service.d/igpu.conf
[Service]
Environment=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
Environment=__GLX_VENDOR_LIBRARY_NAME=mesa
```

Verified afterwards: the process loads only `libEGL_mesa` + `libgallium`, and
`libcuda` no longer appears in its memory map.

## What the 8GB card would do

The 5060 Ti ships as 8GB and 16GB off the same GB206 die, with the same cores,
clocks and 448 GB/s bandwidth — so holding 7899 MiB here reproduces the 8GB
part's budget. Why that substitution is valid and how to repeat it:
[spill-cost.md](../docs/spill-cost.md).

Held to 7799 MiB free, context 4096:

| `-ngl` | Layers on GPU | Result | Prefill | Generate |
|---|---|---|---|---|
| 999 | all 65 | **LOAD_FAIL** | — | — |
| 32 | 32 | **LOAD_FAIL** | — | — |
| **28** | 28 of 65 | PASS | 391.21 tok/s | **5.11 tok/s** |
| 24 | 24 of 65 | PASS | 367.60 tok/s | 4.63 tok/s |

Against the same model fully resident on the 16GB card:

| | 16GB — all 65 layers | 8GB — 28 of 65 layers |
|---|---|---|
| Generate | ~26 tok/s | **5.11 tok/s** |
| Prefill | ~900 tok/s | 391 tok/s |
| Max context | 34,816 | 4,096 tested |

A 27B at IQ4_XS does not fit an 8GB card in any usable sense. Peak VRAM figures
above include the 7899 MiB held by the simulating process.

## VRAM behaviour during inference

Sampled `nvidia-smi` every 50 ms across a full prefill + decode cycle, 481 samples:

```
min 15845 MiB, max 15845 MiB
```

**llama.cpp's VRAM usage is completely static.** All buffers are allocated at
load time and nothing grows during inference. So a config that loads and passes
a full-window prompt will not OOM later from inference alone — the margin only
has to survive other processes touching the card.

## Rejected: NVFP4

Both NVFP4 builds of this model are far too large for 16 GB, because "NVFP4" is
mixed precision rather than 4-bit throughout:

| Build | Size | Notes |
|---|---|---|
| [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | 21.94 GB | attention + `linear_attn` layers are FP8 |
| [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 23.44 GB | same, plus the vision tower left at BF16 |
| GGUF IQ4_XS | **15.44 GB** | ~4.25 bpw across essentially everything |

Only the MLP tensors are 4-bit in either build. The hardware is capable —
5060 Ti is Blackwell sm_120 with native FP4 — but vLLM cannot offload weights to
CPU, so oversized means it simply doesn't start.
