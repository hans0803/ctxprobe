# Gotchas

**English** · [繁體中文](gotchas.zh-TW.md)

Things that quietly cost you context, and how to check for each one.

## 1. `nvidia-smi` total is not your budget

A 16 GB card reports 16311 MiB but CUDA can only ever allocate 15849 MiB. The
**462 MiB gap is driver reserve** — framebuffer, page tables, allocator
structures. It is never available to any process.

```bash
python3 -c 'import torch; print(torch.cuda.mem_get_info()[1] // 2**20, "MiB allocatable")'
```

Headroom measured against `nvidia-smi`'s total reads ~460 MiB more optimistic
than reality, which at ~34 MiB per 1K tokens is a phantom 13K of context.

A second symptom: once a model is loaded with a few hundred MiB "free", a second
process cannot even start, because **a CUDA context alone costs ~138 MiB**.

## 2. `llama-server` defaults to 4 slots

`-c N` is per slot. With the default `--parallel 4`, `-c 8192` allocates KV for
32,768 tokens.

```
load_model: initializing, n_slots = 4, n_ctx_slot = 8192
```

Check `n_slots` in the startup log. For single-user testing, `--parallel 1` is
free context — it took our test card from 4,096 to 16,384 with no other change.

This has changed across llama.cpp versions — some treat `-c` as the total and
divide it among slots. Don't trust this page over your own startup log: read
`n_ctx_slot` and `n_slots` and multiply.

## 3. `n_ctx` is padded to 256

```c
// src/llama-context.cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

`-c 35000` becomes 35072. 256 is the real search granularity; stepping by 1024
skips three testable sizes each step. Confirm with `n_ctx_slot` in the log.

## 4. A short prompt does not qualify a context size

The big one. A longer prompt instantiates CUDA kernels a short one never
touches, and loading one for the first time needs device memory that a
nearly-full card doesn't have. So this happens:

| Context | Loads | 18-token prompt | 64-token prompt |
|---|---|---|---|
| 34816 | yes | 25.99 tok/s | works |
| 35072 | yes | 25.98 tok/s | **CUDA OOM** |

Nothing about the load, the health check, or a small generation distinguishes
these two. **64 tokens does** — the threshold is far lower than "fill the
window", which matters because it makes checking cheap. Confirm the mechanism
with `CUDA_MODULE_LOADING=EAGER`: it loads every kernel up front, turning the
runtime crash into a startup failure you can't miss.

64 comes from `MMQ_DP4A_MAX_BATCH_SIZE`, but note the guard around it:

```c
return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
```

On a card without FP16 MMA the left side short-circuits and everything uses
dp4a, so there is no switch at 64 at all. The threshold is a property of your
GPU generation as much as of llama.cpp.

This is what `ctxprobe` reports as `PREFILL_OOM`, with `died@64` naming the rung
that killed it. It used to be called `LONG_OOM`, which was a leftover from
believing the buffers grew with prompt length — they don't, and the name sent
people looking for a memory leak that isn't there.

**64 is not the only rung that fires.** The ladder climbs 64 → 512 → 4096 for a
reason: 512 is `n_ubatch`, the first full micro-batch, and 4096 clears `n_batch`
(2048). All three have now killed a real config:

| Config | Died at | Rung |
|---|---|---|
| Qwen3.6-27B `IQ4_XS` q8_0 35,072 | `died@64` | `MMQ_DP4A_MAX_BATCH_SIZE` |
| Qwen3.8-27B `UD-IQ4_XS` q8_0 62,208 | `died@509` | `n_ubatch` |
| Qwen3.8-27B `UD-Q4_K_S` f16 19,968 | `died@4024` | `n_batch` |

The last one cleared 64 **and** 512 and died on 4,024 tokens. A check that
stopped at either would have called it a pass. The 512 failures also span both
KV types and therefore both flash-attention states — `q8_0` pins flash attention
on, `f16` leaves it at `auto` — and the failure point does not move, which is
what puts the blame on ggml's memory pool growing against batch shape rather
than on the attention implementation.

Related: when the child dies this way it becomes a defunct process, and a
supervising gateway that only tracks its own state will keep reporting the
deployment as healthy. Check the process, not the status endpoint.

## 5. PEAK_VRAM cannot tell you what will fail

The column everyone reads first is the one that can't answer the question:

| Context | Peak VRAM | Verdict |
|---|---|---|
| 34,816 | 15845 MiB | PASS |
| 35,072 | 15847 MiB | **CUDA OOM** |

Two MiB apart, opposite outcomes. This isn't a sampling problem — it's
structural. The allocation that fails never appears in usage *because it
failed*, and what it was asking for is well under `nvidia-smi`'s 1 MiB
resolution anyway.

Read peak VRAM as "how much headroom is left", never as "how close to failing".
The verdict column is the only one that answers that.

## 6. Thinking models return empty content

Qwen3.6 and similar reasoning models put everything in `reasoning_content`.
With a 1200-token budget the model was still thinking when it hit the limit, so
`content` came back as an empty string and `finish_reason` was `length`:

```json
{"message": {"content": "", "reasoning_content": "Here's a thinking process:..."},
 "finish_reason": "length"}
```

Either give a much larger budget or disable thinking:

```bash
llama-server ... --reasoning-budget 0
# or per request: {"chat_template_kwargs": {"enable_thinking": false}}
```

Benchmarks that only look at tokens/s won't notice; ones that check output will.

## 7. `n_ubatch` 512 is a bad default for sparse MoE

Prefill reads experts once per ubatch. With top-6-of-256 routing, a 512-token
batch collectively touches nearly every expert, so **each ubatch drags the whole
expert set of a layer across the bus** — a cost amortised over however many
tokens share that batch.

Raising it on a DeepSeek-V4-Flash (137 GB, experts in DDR5):

| ubatch | Prefill | VRAM |
|---|---|---|
| 512 (default) | 151 tok/s | 9.1 GB |
| **8192** | **775 tok/s** | 12.5 GB |

**5.1× for 3.5 GB.** Past 8192 prefill is flat while VRAM keeps climbing, so
bigger is not automatically better — aim for "splits your prompt into 2-3
passes".

Two caveats. The gain is zero below 512 tokens (one pass either way), so short
chat turns see nothing. And generation is unaffected — decoding is batch-of-1,
which is also what makes this measurable cleanly.

This only matters when experts are on the far side of a slow link. A dense model
fully in VRAM has no equivalent cliff.

## 8. `--tensor-split` cannot fix uneven MoE placement

The instinct when one card fills up and the other sits idle is to reach for
`--tensor-split`. On a MoE model with experts offloaded, it does nothing:

```
--n-cpu-moe 35, two 32 GB cards
  GPU0  8.4 GB      GPU1  31.1 GB     ← 24 GB idle
--tensor-split 1,1 and 3,1 → both OOM
```

Two mechanisms compose badly. Layer split is **contiguous** — one card takes an
early block of layers, the other takes the rest. `--n-cpu-moe N` leaves the
GPU-side experts in the **last** N layers, also contiguous. That block lands
whole on one card, and moving the split point cannot separate it.

The fix is `-ot`, naming tensors explicitly:

```bash
-ot "blk\.(3[0-5])\.ffn_.*_exps\.weight=CUDA0" \
-ot "blk\.(3[6-9]|4[0-2])\.ffn_.*_exps\.weight=CUDA1" \
--cpu-moe
```

**One regex range per flag, and `-ot` before `--cpu-moe`.** Two ways to get it
silently wrong: putting a `=CPU` pattern first (it swallows everything), or
joining patterns with commas (the greedy `.*` eats the separator and matches
nothing). Neither errors — the tell is VRAM sitting at the no-experts-on-GPU
baseline while generation stays at the all-CPU figure.

Worth knowing before buying a second card: doing this correctly bought a measured
+23% prefill and **no demonstrable generation gain at all**, against 5.1x prefill
from raising `n_ubatch` alone.

## 9. Desktop processes squat on the discrete GPU

Two separate offenders, worth ~590 MiB together on our box:

- **Xorg / GNOME Shell** — fixed by pointing the display at an integrated GPU
  (BIOS: `Primary Display = IGFX`, `iGPU Multi-Monitor = Enabled`).
- **GNOME Remote Desktop** — keeps ~130 MiB on the dGPU *even after* the display
  moves, because it loads `libcuda` + `libnvidia-encode` for NVENC.

```ini
# ~/.config/systemd/user/gnome-remote-desktop.service.d/igpu.conf
[Service]
Environment=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
Environment=__GLX_VENDOR_LIBRARY_NAME=mesa
```

Verify anything that claims to have moved:

```bash
grep -cE 'libcuda|libnvidia-encode' /proc/<pid>/maps   # want 0
```

## 10. A short generation is not a tok/s measurement

The mirror of #4. A generation long enough to exercise the decode path is not
long enough to time it.

Sweeping expert placement for DeepSeek-V4-Flash, sampling 32 generated tokens
per configuration:

| Expert layers on GPU | Generate |
|---|---|
| 13 | 15.86 |
| 14 | 13.70 |
| 15 | 14.34 |
| 16 | 13.38 |

More VRAM, less speed, no ordering — a table that demands an explanation, and
one is easy to invent (routing is per-token, so surely throughput varies with
what the model happens to generate). Three repeats at **256** tokens on the same
hardware:

```
12.09   12.04   12.07      <- ±0.2%
```

The scatter was the sample. At 32 tokens, per-request setup and the first-token
path are still a visible fraction of a window that llama.cpp averages over
whole, and the residual is the same size as the effect being measured.

The cost is not the wasted run. It is that a noisy table looks like a finding,
and the explanation you reach for will feel mechanistic and survive review —
[the DeepSeek-V4-Flash writeup](../results/rtx5090-32gb-deepseek-v4-flash.md)
carried exactly that error for a day. Prefill is not affected the same way; it
is timed over thousands of tokens by construction.

Use 256 tokens and at least three repeats before reading anything into a
difference under 10%.

---

Scope note: this list covers things that consume or waste **GPU memory on a
single card** — that's the question ctxprobe answers. What it costs to spill
past the card is measured separately in [spill-cost.md](spill-cost.md).
Multi-GPU placement is out of scope; `llama-fit-params` handles it.

See also: [model-quant.md](model-quant.md) for picking a quant that fits, and
[kv-cache-quant.md](kv-cache-quant.md) for halving the cache that grows with
context.
