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

## 3. `n_ctx` is padded to 256

```c
// src/llama-context.cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

`-c 35000` becomes 35072. 256 is the real search granularity; stepping by 1024
skips three testable sizes each step. Confirm with `n_ctx_slot` in the log.

## 4. A short prompt does not qualify a context size

The big one. Prefill compute buffers scale with prompt length, so this happens:

| Context | Loads | Short prompt | Full-size prompt |
|---|---|---|---|
| 34816 | yes | 25.99 tok/s | works |
| 35072 | yes | 25.98 tok/s | **CUDA OOM** |

Nothing about the load, the health check, or a small generation distinguishes
these two. Only a prompt that fills the window does.

Related: when the child dies this way it becomes a defunct process, and a
supervising gateway that only tracks its own state will keep reporting the
deployment as healthy. Check the process, not the status endpoint.

## 5. Thinking models return empty content

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

## 6. Desktop processes squat on the discrete GPU

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

---

Scope note: this list covers things that consume or waste **GPU memory on a
single card** — that's the question ctxprobe answers. What it costs to spill
past the card is worth measuring for contrast, but it's a different list.
Multi-GPU placement is out of scope; `llama-fit-params` handles it.
