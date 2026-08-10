# ctxprobe

**English** · [繁體中文](README.zh-TW.md)

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
  kv cache  : q8_0 | slots: 1 | step: 256 | long prompt fills 95%

CONTEXT    RESULT      PEAK_VRAM   PREFILL    GENERATE   FILLED
------------------------------------------------------------------------
32768      PASS        15767       867.97     24.76      31057
36864      DECODE_OOM  15845       n/a        n/a        —
34816      PASS        15845       858.39     24.62      32997
35840      DECODE_OOM  15805       n/a        n/a        —
35328      DECODE_OOM  15787       n/a        n/a        —
35072      LONG_OOM    15847       89.48      26.36      —

Largest context that actually runs: 34816 tokens
  peak VRAM 15845 MiB | prefill 858.39 tok/s | generate 24.62 tok/s
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

35072 loads fine and generates at full speed. Then a realistic prompt kills it:

```
launch_mul_mat_q at mmq.cuh:1375
cudaFuncSetAttribute(mul_mat_q<type, J, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)
CUDA error: out of memory
```

**A long prompt reaches CUDA kernels a short one never touches.** llama.cpp's
quantized matmul is templated on batch shape, so a larger prefill instantiates a
different `mul_mat_q` variant — and under CUDA's lazy module loading (the default
since CUDA 12), touching a kernel for the first time loads its code into device
memory. With VRAM nearly exhausted, that load is what fails.

Note what this is *not*: allocation growing during inference. Sampled every
50 ms across a full prefill and decode, VRAM never moved off 15845 MiB. The run
simply needs memory to load a kernel it hasn't loaded yet.

Two experiments pin this down. Forcing kernels to load up front turns the
mysterious runtime crash into a deterministic startup failure:

```
$ CUDA_MODULE_LOADING=EAGER llama-server ... -c 35072
allocating 251.53 MiB on device 0: cudaMalloc failed: out of memory
llama_init_from_model: failed to allocate compute pp buffers
```

And the lethal prompt is far shorter than "fills the window". Climbing prompt
lengths against both configs:

| Prompt | 34,816 (passes) | 35,072 (fails) |
|---|---|---|
| 16 tokens | ok | ok |
| 32 tokens | ok | ok |
| **64 tokens** | ok | **dies** |
| 2,025 tokens | ok | — |

**64 tokens is enough to separate them.** ctxprobe's own warmup prompt is 18
tokens, which is exactly why it needed a long prompt to notice — not because
filling the window matters, but because 18 tokens sits below the threshold where
a new kernel variant gets instantiated. The prompt ladder exploits this: it
climbs 64 → 512 → 4096 → full and stops at the first death, so a doomed config
is rejected in seconds instead of after a 30K-token prefill.

`PASS` therefore means the run survived a prompt filling **95% of the window**
and still emitted tokens. That percentage is measured, not estimated: the length
is converged on using the server's own `/v1/chat/completions/input_tokens`
endpoint, so it accounts for the chat template wrapper — which is exactly what
tips a near-full prompt over the limit. The `FILLED` column reports the real
prompt size that was pushed through.

Only 8 tokens are requested back. The question is whether prefill at that depth
survives and the model still speaks, not how fast it writes.

## Start here

Two settings decide most of the outcome on a single card:

- **[Which quant should you download?](docs/model-quant.md)** — reading
  `Q4_K_M` / `IQ4_XS` names, and the rule that matters most: the biggest quant
  that *fits entirely* beats a better one that spills.
- **[KV cache: why q8_0](docs/kv-cache-quant.md)** — the highest-leverage
  setting for long context. Worth 72% more context here, for one flag.
- **[What it costs when it doesn't fit](docs/spill-cost.md)** — measured: 5×
  slower for spilling 57% of the layers, and how to test a card you don't own.

Then [gotchas.md](docs/gotchas.md) for what silently eats VRAM, and
[results/](results/) for measured configurations.

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

Three more are in [gotchas.md](docs/gotchas.md): thinking models returning empty
output, desktop processes squatting on the card, and dead children still being
reported healthy.

## Usage

```
ctxprobe MODEL.gguf [options] [-- extra llama-server args]

  --min N        lower bound (default 2048)
  --max N        upper bound (default: model's trained context, capped 131072)
  --step N       granularity (default 256)
  --kv TYPE      KV cache type: q8_0 (default), f16, q4_0
  --parallel N   server slots (default 1)
  --list "A B C" test these sizes instead of binary-searching
  --fill PCT     how full the long prompt should be (default 95)
  --ngl N        layers on GPU (default 999 = all; lower spills to system RAM)
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

| GPU | Model | Quant | KV | Max context | Prefill | Generate |
|---|---|---|---|---|---|---|
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | q8_0 | 34,816 | 858 tok/s | 24.6 tok/s |
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | f16 | 20,224 | 923 tok/s | 26.9 tok/s |
| RTX 5060 Ti 8GB *(simulated)* | Qwen3.6-27B | IQ4_XS | q8_0 | 4,096 † | 391 tok/s | 5.11 tok/s |

† Does not fit: only 28 of 65 layers on the GPU, the rest in system RAM. See
[spill-cost.md](docs/spill-cost.md).

The first two rows are the same model on the same card, one flag apart. Full run,
including how the ceiling moved as VRAM was freed:
[rtx5060ti-16gb-qwen3.6-27b.md](results/rtx5060ti-16gb-qwen3.6-27b.md).

Speeds come from a window-filling prompt, so they describe a loaded window
rather than an empty one. Don't read the f16 row as "f16 generates faster" —
it sits at 20K context against 34K, and generation slows as the window fills.

Contributions for other cards welcome; `--json` output is meant to be pasted
straight in.

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
