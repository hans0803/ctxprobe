# ctxprobe

**Estimators tell you a context size should fit. ctxprobe proves it runs.**

Finds the largest context window a GGUF model *actually* works at on a single
GPU — by booting it, filling the window with a real prompt, and checking it
survives. One bash script, no dependencies beyond `llama-server` and `python3`.

```bash
# with no bounds it searches from 2048 up to the model's trained context;
# narrowing the range just saves boots
./ctxprobe ~/models/Qwen3.6-27B-IQ4_XS.gguf --min 32768 --max 36864 -- --reasoning-budget 0
```

Real output, RTX 5060 Ti 16GB:

```
  model     : Qwen3.6-27B-IQ4_XS.gguf (15G)
  gpu       : NVIDIA GeForce RTX 5060 Ti
  vram      : 16311 MiB reported by nvidia-smi, 15 MiB already in use
              15849 MiB actually allocatable (462 MiB is driver reserve)
  kv cache  : q8_0 | slots: 1 | step: 256 | long prompt fills 75%

CONTEXT    RESULT      PEAK_VRAM   PREFILL    GENERATE
------------------------------------------------------------
32768      PASS        15767       907.11     24.46
36864      DECODE_OOM  15845       n/a        n/a
34816      PASS        15845       900.82     24.29
35840      DECODE_OOM  15805       n/a        n/a
35328      DECODE_OOM  15787       n/a        n/a
35072      LONG_OOM    15847       88.81      26.41

Largest context that actually runs: 34816 tokens
  peak VRAM 15845 MiB | prefill 900.82 tok/s | generate 24.29 tok/s
```

Six boots to bracket the ceiling. Note the last row: 35072 loaded, decoded a
short prompt at full speed, and still died on a real one — that's the row no
load-time estimate can produce.

## Why this exists

There are already good VRAM *estimators* — [oobabooga's regression formula](https://oobabooga.github.io/blog/posts/gguf-vram-formula/)
(19,517 measurements), llama.cpp's own `llama-fit-params`, and half a dozen
calculators. They answer **"will it load?"** and they answer it well.

That is not the same question as **"will it work?"**

Real numbers from an RTX 5060 Ti (16 GB), Qwen3.6-27B-IQ4_XS + q8_0 KV:

| Context | Loads? | Short prompt | 30K-token prompt |
|---|---|---|---|
| 34816 | yes | 25.99 tok/s | **works** |
| 35072 | yes | 25.98 tok/s | **CUDA OOM, server dies** |

35072 loads fine and generates at full speed. Then a realistic prompt kills it.
**Prefill compute buffers scale with prompt length**, and no load-time estimate
models that — so anything validated with a short prompt reports a false pass.
ctxprobe's `PASS` requires surviving a prompt that fills most of the window.

## Three things that will cost you VRAM

Found the hard way; all three are invisible to estimators.

**1. `nvidia-smi` overstates your VRAM.** On a 16 GB card, 462 MiB is driver
reserve that no allocation can ever touch:

```
nvidia-smi total          16311 MiB
CUDA can allocate         15849 MiB   <- the real ceiling
```

Compute headroom against `torch.cuda.mem_get_info()[1]`, not `nvidia-smi`.
ctxprobe prints both.

**2. `llama-server` defaults to 4 slots.** `-c 8192` reserves KV for
**4 × 8192 = 32768** tokens. Passing `--parallel 1` quadrupled the usable
context on our test card (4096 → 16384) before anything else was tuned.
ctxprobe defaults to `--parallel 1`.

**3. `n_ctx` is padded to a multiple of 256.** From `llama-context.cpp`:

```c
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

So `-c 35000` silently becomes 35072. Searching in steps of 1024 (a natural
habit) skips three testable points every step; ctxprobe's default step is 256.

## Usage

```
ctxprobe MODEL.gguf [options] [-- extra llama-server args]

  --min N        lower bound (default 2048)
  --max N        upper bound (default: model's trained context, capped 131072)
  --step N       granularity (default 256)
  --kv TYPE      KV cache type: q8_0 (default), f16, q4_0
  --parallel N   server slots (default 1)
  --list "A B C" test these sizes instead of binary-searching
  --fill PCT     how full the long prompt should be (default 75)
  --json         machine-readable output
  --keep         keep per-run logs
```

Binary search by default, so finding a ceiling takes ~log₂(range) boots rather
than one per candidate. Each boot reloads the model, so expect a few minutes.

Extra `llama-server` flags pass through after `--`:

```bash
# Qwen3.6 and other thinking models burn the whole budget on reasoning and
# return empty content unless you disable it
./ctxprobe model.gguf -- --reasoning-budget 0
```

`LLAMA_SERVER=/path/to/llama-server` overrides binary discovery.
`CUDA_VISIBLE_DEVICES` picks the GPU (single-GPU only by design).

Reporting the real allocatable total needs `torch` importable from some Python
on the box (conda envs are searched automatically; `CTXPROBE_PYTHON` points at a
specific one). It has to be `cudaMemGetInfo` — NVML and `nvidia-smi` report the
board's physical size, which is exactly the number that hides the reserve.
Without torch everything still works, you just don't get that line.

## Results

Measured configurations live in [results/](results/). Contributions for other
cards welcome — `--json` output is meant to be pasted straight in.

| GPU | Model | Quant | KV | Max context | Generate |
|---|---|---|---|---|---|
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | q8_0 | 34,816 | 26.0 tok/s |

## Scope

**One card, one number: how much context its VRAM actually sustains.**

A `PASS` means the whole model ran on the GPU — `-ngl 999`, no CPU offload — so
host RAM stays out of the budget and the ceiling is attributable to the card
alone.

Spilling past the card isn't off the table, it's just not a pass. Measuring
what overflow actually costs is the clearest way to show why staying in VRAM
matters, so those runs belong here too — reported as a labelled comparison
rather than as a working configuration.

Multi-GPU placement is out of scope; `llama-fit-params` handles it well.

## License

MIT
