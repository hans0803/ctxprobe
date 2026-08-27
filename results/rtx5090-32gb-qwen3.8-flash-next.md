# RTX 5090 32GB — Qwen3.8-Flash-Next (UD-IQ4_XS)

**English** · [繁體中文](rtx5090-32gb-qwen3.8-flash-next.zh-TW.md)

Measured 2026-08-27. A 176.9 B-parameter model, 87 GiB on disk, on one 32 GB
card. That it fits is not the news — [DeepSeek-V4-Flash
](rtx5090-32gb-deepseek-v4-flash.md) already showed a 137 GB model doing that.
The news is that **29% of this model is a hash table that costs 1.4 KB of weights
per token**, and what happens when you try to take it off the RAM budget.

> ⚠️ **This run uses unmerged code.** `qwen4exp` support is not in llama.cpp
> master; it lives in [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742),
> built here at `213df58`. One later experiment adds a one-line env-gated patch to
> `llama-mmap.cpp`, off by default, shown where it is used.
>
> **Not measured yet:** the context ceiling. Speed figures are at the default
> `n_ubatch` 512 and f16 KV unless the section says otherwise.

## Hardware

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5090 — 32607 MiB, one card via `CUDA_VISIBLE_DEVICES=1`** |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads, single NUMA node |
| RAM | 186 GB DDR5, dual channel |
| NVMe | Kingston SFYRD2000G (PCIe 4.0), ext4, `read_ahead_kb` 128 |
| llama.cpp | PR #27742 @ `213df58`, build 861, CUDA 12.8 (`sm_120`) |
| Model | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ4_XS`, 87.24 GiB, 3 shards |

The second 5090 serves an unrelated vLLM instance throughout; every figure here
is single-card.

Before trusting the build, `test-llama-archs -a qwen4exp` was run on both
backends: **CUDA OK at NMSE 8.51e-08, CPU OK at 0.00e+00**, matching the PR's
own numbers. The `Meta` backend asserts (`ggml-backend-meta.cpp:756`) — a real
bug in the branch, on the graph-analysis path that an explicit `-ngl` skips.

## Architecture

```
qwen4exp.block_count                = 48
qwen4exp.expert_count               = 512
qwen4exp.expert_used_count          = 10       <- 512 choose 10
qwen4exp.expert_feed_forward_length = 640
qwen4exp.embedding_length           = 2560
qwen4exp.full_attention_interval    = 4        <- 12 full, 36 gated delta-net
qwen4exp.attention.indexer.top_k    = 2048     <- QSA sparse attention
qwen4exp.ple.layers                 = [1]
qwen4exp.context_length             = 262144
```

Where the bytes are, from ggml block sizes rather than the label:

| | Params | GiB | bpw |
|---|---|---|---|
| **MoE experts** | 120,795,955,200 | **55.43** | 3.942 |
| **Engram (n-gram table)** | 51,200,245,760 | **26.82** | 4.500 |
| everything else | 3,669,736,320 | 3.86 | 9.042 |
| embed / output | 1,277,962,240 | 1.12 | 7.536 |
| **TOTAL** | 176,943,899,520 | **87.24** | 4.235 |

## The engram: 29% of the model, 1.4 KB per token

`per_layer_token_embd.weight` is one tensor of **(160, 320,001,536)** at
`IQ4_NL` — 51.2 B parameters, 26.82 GiB — and it exists **only at layer 1**
(`ple.layers = [1]`). "Per-layer embeddings" names where it is injected, not how
many copies there are.

The 320 M rows are 16 head regions of ~20,000,0xx rows each, where
16 = `(ngram_size 3 − 1) × heads_per_ngram 8`: eight bigram heads and eight
trigram heads. Row selection is a host-side hash, because ggml has neither int64
nor xor:

```
mixed_n = (t[p]·m0) ⊕ (t[p−1]·m1) ⊕ … ⊕ (t[p−n+1]·m_{n−1})     n = 2, 3
row_h   = mixed_n mod vocab[h] + offset[h]
```

The multipliers are in the file (`ple.layer_multipliers =
[23703573157769, 20109073645365, 8052911324071]`); an EOS resets the window.

Each token gathers 16 rows of 160 values — **1,440 bytes of weights, 16 pages,
once per forward pass** — then a key/value projection and a sigmoid gate against
the residual stream decide how much of it to inject. Against the experts' 1.16 GB
per token that is a factor of 800,000, which is why the table deserves separate
treatment from the rest of the file.

It is not portable. The rows are trained under a specific hash collision pattern
(trigram space 248,320³ into 20 M rows per head, ~7.6 × 10⁸ trigrams per row);
change the tokenizer, the multipliers, or the vocab sizes and the table is noise.

## Expert placement decides generation speed

`-ncmoe N` keeps the experts of the first N layers on the CPU. `-c 8192`,
`--threads 16`, 128-token generations, warm:

| Layers on CPU | On GPU | Peak VRAM | Generate | ms/token |
|---|---|---|---|---|
| 48 | 0 | 6,487 | 24.98 | 40.03 |
| 40 | 8 | 16,387 | 28.89 | 34.61 |
| 34 | 14 | 23,215 | 31.88 | 31.37 |
| 30 | 18 | 28,161 | 34.40 | 29.07 |
| **28** | **20** | **30,435** | **35.53** | **28.15** |
| 26 | 22 | — | — | **load fails** |

**20 layers is the ceiling on this card.** Each layer's experts cost ~1,192 MiB
of VRAM (from the deltas; 55.43 GiB / 48 predicts 1,182), so 22 layers needs
32,711 MiB against 32,607.

Least squares over the five passing points:

```
t_token = 11.32 ms + 28.44 ms × (layers_on_cpu / 48)

residuals all under 0.7 ms
```

**28.44 ms is the DDR5 expert read**, and it is nearly optimal. Per token the CPU
reads 48 × 10 × 4,915,200 params at 3.942 bpw = **1.1625 GB**:

```
1.1625 GB / 28.44 ms = 40.9 GB/s
```

against what this box sustains, measured directly:

| Threads | Sequential read |
|---|---|
| 1 | **50.3 GB/s** |
| 4 | 46.2 |
| 8 | 45.7 |
| 16 | 44.5 |
| 32 | 43.8 |

**40.9 of 44.5 is 92%.** The expert path is bandwidth-bound. A single core nearly
saturates dual-channel DDR5, so more threads cannot help and actively hurt:

| Threads | Generate (`-ncmoe 48`) |
|---|---|
| **16** | **25.10** |
| 24 | 23.90 |
| 28 | 23.72 |

**11.32 ms is the floor** — the GPU graph, attention, delta-net, QSA indexer, PLE
gate — and no VRAM removes it. It caps this model in this build at **88.3 tok/s**,
reachable only with all 48 expert layers resident: 6,487 + 48 × 1,192 = 63.7 GB.
Two 5090s, not one.

## The card is not the bottleneck

At `-ncmoe 48`, sampled during generation:

| | |
|---|---|
| GPU utilisation | **19–20%** |
| GPU memory-controller | 7–8% |
| GPU power | 113 W of ~575 W |
| CPU | **50.1% user of 32 threads** — 16 saturated |
| VRAM | 6,487 MiB of 32,607 |

**A 176.9 B model runs on 6.5 GB of VRAM with the card 80% idle.** Moving expert
layers onto it converts idle silicon into 42% more throughput.

## Prefill: `-ub`, and where the VRAM has to come from

Same mechanism as [DeepSeek-V4-Flash](rtx5090-32gb-deepseek-v4-flash.md): a
512-token ubatch under 512-choose-10 routing touches nearly every expert in a
layer, so each ubatch reads the whole layer off DDR5 and that cost is amortised
over however many tokens are in the batch. Raising `-ub` is the lever. But the
compute buffer is sized by `-ub` at load, and on this card the VRAM is already
spoken for.

15,505-token real-prose prompt, `-c 24576`, `-b` = `-ub`, `--threads 16`:

**At `-ncmoe 28` — the best decode placement — nothing above 512 fits.**

| ub | Peak VRAM | Prefill | TTFT | Generate | |
|---|---|---|---|---|---|
| 512 | 31,963 | 254 | 61.0 s | 31.79 | PASS |
| 1024 | 32,101 | — | — | — | PREFILL_OOM |
| 2048 | 32,103 | — | — | — | PREFILL_OOM |
| 4096 | — | — | — | — | LOAD_OOM |
| 8192 | — | — | — | — | LOAD_OOM |

Sixty-one seconds to the first token of a 15K prompt, and 146 MiB of headroom
to do anything about it.

**At `-ncmoe 36` — eight layers back on the CPU — the curve appears.**

| ub | Peak VRAM | Prefill | TTFT | Generate | vs 512 |
|---|---|---|---|---|---|
| 512 | 22,471 | 202 | 76.6 s | 27.75 | — |
| 1024 | 22,755 | 323 | 47.9 | 27.68 | 1.6× |
| 2048 | 23,025 | 520 | 29.8 | 27.49 | 2.6× |
| 4096 | 23,567 | 790 | 19.6 | 27.77 | 3.9× |
| **8192** | **25,453** | **1,040** | **14.9** | 27.66 | **5.1×** |

**5.1× — the same multiplier DeepSeek gave**, for 2,982 MiB of compute buffer
(DeepSeek's was 3.5 GB). Generation is flat at 27.5–27.8 the whole way, which
is the control that makes the prefill column unambiguous.

The trade on one card, then:

| | `-ncmoe 28`, ub 512 | `-ncmoe 36`, ub 8192 |
|---|---|---|
| Generate | **31.8** | 27.7 (−13%) |
| Prefill | 254 | **1,040 (4.1×)** |
| TTFT, 15.5K prompt | 61 s | 15 s |

Eight expert layers moved off the card buy 3 GB, and 3 GB buys four times the
prefill for 13% of the decode. Which side of that you want depends entirely on
prompt length; for anything with a system prompt and history it is the right
side.

## Two cards

The engram stays in RAM whatever happens (it never left it in any run above).
Everything else is 87.24 − 26.82 = **60.42 GiB**, and two 5090s are nominally
64 GB. So does it fit?

With the vLLM instance taken down for the duration, `CUDA_VISIBLE_DEVICES=0,1`,
`-c 8192`, `--threads 16`:

| Config | GPU 0 | GPU 1 | Generate | ms/token | |
|---|---|---|---|---|---|
| `-ncmoe 0`, default split | — | — | — | — | **load fails, 66 MiB short on GPU 0** |
| `-ncmoe 2`, default split | 30,621 | 30,721 | 82.77 | 12.08 | PASS |
| `-ncmoe 2`, `--tensor-split 25,23` | 31,845 | 29,495 | 83.72 | 11.94 | PASS |
| `-ncmoe 4`, `--tensor-split 26,22` | 30,633 | 28,273 | 73.02 | 13.69 | PASS |
| `-ncmoe 0`, `--tensor-split 25,23` | — | — | — | — | load fails |

**Not quite — but two layers on the CPU gets 83 tok/s, 2.36× the single card.**
And the law from the single-card sweep held across the PCIe boundary without
adjustment: `-ncmoe 4` was predicted at 13.69 ms and measured at 13.69;
`-ncmoe 2` predicted 12.5 and came in at 11.94–12.08. The last two layers are
4% of the expert bytes and worth about 1.2 ms a token — going from 83 to the
88 tok/s floor is a 6% gain, not the leap "all on GPU" suggests.

### Why 60 GiB does not fit in 64 GB

Three things the nominal figure leaves out, all measured:

1. **Driver reserve.** Each card reports 32,607 MiB and allocates 32,109
   (498 MiB reserved, from the DeepSeek page). Two cards: 64,218 MiB, 62.7 GiB.
2. **Per-card overhead.** CUDA context, compute buffer and KV are paid on
   *each* card. Single-card, `-ncmoe 48` used 6,487 MiB of which 5,100 was
   weights — 1,387 MiB of overhead at `-c 8192`. Two cards: ~2,400.
3. **Layer granularity.** The split can only move in whole layers, 1,274 MiB
   each, so one card always carries the rounding.

```
allocatable      64,218
weights          61,870   (60.42 GiB)
per-card overhead ~2,400
                 ───────
                   ~ −50 MiB
```

It is not a total; it is GPU 0, alone, 66 MiB over — that being the size of
the allocation that failed, so the true gap is *at least* 66.

**q8_0 KV does not close it.** Twelve full-attention layers with two KV heads
is 24 KiB a token at f16 and ~14.8 at q8_0 (with its per-layer scratch —
see [kv-cache-quant.md](../docs/kv-cache-quant.md)). At `-c 8192` that saves
74 MiB across both cards, 37 on the one that is short. Untested, but the
arithmetic is not close.

**A smaller quant does.** From the GGUF headers of the other two builds in the
repo, fetched without downloading them:

| | File | Expert bpw | Non-engram | Per card at `-ncmoe 0` | Margin |
|---|---|---|---|---|---|
| `UD-IQ4_XS` (this run) | 87.2 GiB | 3.942 | 60.42 GiB | 32,135 | **over** |
| **`UD-Q3_K_XL`** | 83.8 | 3.697 | **56.97** | 30,369 | **1.7 GB** |
| `UD-IQ3_XXS` | 76.3 | 3.220 | 49.50 | 26,544 | 5.5 GB |

The engram is `IQ4_NL` at 26.82 GiB in every one of them — unsloth pins it —
so all of the reduction lands on the experts, which is exactly the part that has
to fit. `UD-Q3_K_XL` gets `-ncmoe 0` onto two cards with 1.7 GB to spare;
`UD-IQ3_XXS` leaves enough for real context or a large `-ub`, at 3.22 bpw on the
experts, which [model-quant.md](../docs/model-quant.md) files under "only when
nothing else fits". There is no `IQ4_XXS`; ggml's IQ4 family is `_XS` and `_NL`.

### The default split, and where `-ncmoe` takes its relief from

The prediction going in — from [gotcha #8](../docs/gotchas.md) — was that
`-ncmoe 2` with the default split would land 24 expert layers on GPU 1 against
22 on GPU 0 and OOM the heavier card. It did not: **at `-ncmoe 2` the default
split came out balanced to within 100 MiB** (30,621 / 30,721), and the explicit
`25,23` unbalanced it by 2,350.

But that balance is not a general property. In the prefill sweep below, GPU 1
sits at exactly the same figure at `-ncmoe 4` as at `-ncmoe 2` — 31,295,
31,763, 32,015 at each `-ub` — while GPU 0 drops by 2,560. **The boundary does
not move with `-ncmoe`, and every layer `-ncmoe` takes off the GPU comes off
GPU 0.** At `-ncmoe 4` the default therefore leaves GPU 0 with 2.6 GB spare and
GPU 1 full — the shape gotcha #8 describes — and it is why `-ub 2048` dies on
device 1 at both settings. Whether `-ncmoe 2`'s balance was the splitter's
doing or a coincidence of that one boundary is not resolved here.

The practical rule: pair `-ncmoe N` with a `--tensor-split` that moves layers
*onto* GPU 0. `-ncmoe 4 --tensor-split 26,22` measured 30,633 / 28,273 above,
which would leave GPU 1 the ~1 GB that `-ub 2048` asks for. Untested in that
combination.

### Two cards: prefill

This is where the second card pays. Same 15,505-token prompt, `-c 24576`,
default split:

| `-ncmoe` | ub | GPU 0 | GPU 1 | Prefill | TTFT | Generate | |
|---|---|---|---|---|---|---|---|
| 2 | 512 | 31,199 | 31,295 | **1,199** | **12.9 s** | 67.3 | PASS |
| 2 | 1024 | 31,657 | 31,763 | **1,527** | **10.2 s** | 66.1 | PASS |
| 2 | 2048 | 32,085 | 32,015 | — | — | — | PREFILL_OOM, 1,006 MiB on device 1 |
| 2 | 4096 | — | — | — | — | — | LOAD_OOM |
| 4 | 512 | 28,639 | 31,295 | 927 | 16.7 s | 59.6 | PASS |
| 4 | 1024 | 28,975 | 31,763 | 1,249 | 12.4 s | 59.9 | PASS |
| 4 | 2048 | 29,317 | 32,015 | — | — | — | PREFILL_OOM, 1,006 MiB on device 1 |
| 4 | 4096 | — | — | — | — | — | LOAD_OOM |

**At the default `-ub` 512, two cards prefill at 1,199 tok/s against one card's
254 — 4.7×, and the 15K prompt's first token arrives in 13 s instead of 61.**
That is before touching the `-ub` lever: with the experts in VRAM the DDR5
amortisation problem mostly disappears, and batch size stops mattering as much.
`-ub` 1024 adds 27%; 2048 does not fit, for the GPU 1 reason above.

Against the single card's best prefill configuration:

| | 1× 5090, `-ncmoe 36`, ub 8192 | 2× 5090, `-ncmoe 2`, ub 1024 |
|---|---|---|
| Prefill | 1,040 | **1,527** |
| TTFT, 15.5K | 14.9 s | **10.2 s** |
| Generate (15.5K filled) | 27.7 | **66.1** |

The two-card box wins both axes at once, which one card cannot — there,
prefill and decode compete for the same 3 GB.

Two more things the table says. Each CPU expert layer costs prefill about
0.12 ms per token at `-ub` 512 (`-ncmoe 2` → `4` is 1,199 → 927), the same
per-layer figure the single-card sweeps gave (254 → 202 for eight layers) —
a property of the DDR5 read, not of the card count. And generation here is 67
rather than the 83 measured at `-c 8192` with a short prompt: 15.5K tokens of
KV in the window cost ~20%, more than the ~10% the single card lost, because
the GPU-side floor that grows with context is a larger share of a 12 ms token
than of a 28 ms one.

Extrapolating the per-layer cost to `-ncmoe 0` gives ~1,700 tok/s at `-ub` 512
— what `UD-Q3_K_XL` would be expected to reach with everything on two cards.
Extrapolation, not measurement.

## Engram on SSD

The question: with experts at `-ncmoe 28` (32.3 GiB of them on the CPU side),
can the 26.8 GiB engram be left on NVMe instead of in RAM?

### There is no per-tensor switch

llama.cpp maps the whole file. Nothing can evict one tensor's pages while the
mapping exists — measured, 512 MiB `MAP_PRIVATE`, fully touched:

| Action | Resident after |
|---|---|
| `posix_fadvise(DONTNEED)` from another fd | **100%** |
| `madvise(MADV_DONTNEED)` on the mapping | **100%** |
| unmap first, then `posix_fadvise(DONTNEED)` | 0% |
| touch under cgroup `MemoryMax=192M` | 37% |

The kernel's `invalidate_mapping_pages()` skips mapped pages. Only reclaim under
memory pressure evicts them, so the experiment is a **whole-process cgroup cap**
that relies on LRU keeping the every-token expert pages and shedding the engram.

`--no-mmap` would silently destroy this: the weights become anonymous memory and
the cap sends them to swap (7 GB here), not back to the GGUF.

### The cap is a blunt instrument, and it shows

Varied prompts (220 words each from a 50-word list, so trigrams are fresh),
13 requests, ~285 prompt + 64 generated tokens each. Cache dropped before every
arm so pages are charged to the right cgroup. Timings are from the 13th request;
disk is `/proc/pid/io` over all 4,472 tokens.

| Cap | File pages | Experts resident? | Generate | Prefill | Disk / token |
|---|---|---|---|---|---|
| none | 87.3 GiB | ✅ | 34.56 | **282** | 0 |
| 46G | 39.2 | ✅ (32.3 needed) | 34.05 | **217** | 3.13 MiB |
| 42G | 35.2 | ✅ | 34.31 | 217 | 4.81 MiB |
| 38G | 31.2 | **❌ thrashing** | 28.64 | 157 | 4.47 MiB |

Read the last row as a failed arm, not a data point: with file pages below the
32.3 GiB the experts need, what is being paid for is expert misses, and the
engram cannot be separated from that.

The two clean arms agree on three things:

**Decode barely notices.** 34.05 and 34.31 against 34.56. During generation the
engram's 16 rows a token are either already cached or 16 faults at ~54 µs, and
16 × 54 µs is 0.9 ms of a 28 ms token. The steady-state run below puts the
real figure at −3.6%.

**Prefill pays.** 217 against 282 at the 13th request. But the 13th request is
not steady state — see the trajectory below, and the real-prose run after it.

**The disk reads are 50× the weights.** 3.13 MiB per token is 200 KiB per row
gathered, for 90 bytes of `IQ4_NL` in a 4 KiB page. `read_ahead_kb` is 128 on
this drive and `llama-mmap.cpp` advises `POSIX_FADV_SEQUENTIAL` on the whole
file, which doubles the readahead window; fault-around adds 64 KiB. 200 KiB per
fault is what that arithmetic gives. Correct for dense weights read once in
order; for a 320 M-row table read at random it drags in 50 pages per useful one,
which is also why the 42G arm reads *more* than 46G — the inflated footprint
churns faster in less slack.

### It is a warm-up curve, not a steady state

Per request, the 46G arm:

| Request | Prefill | Generate |
|---|---|---|
| 0 | 10.8 | 20.6 |
| 1 | 78.9 | 25.9 |
| 2 | 152.3 | 32.0 |
| 3 | 162.9 | 33.5 |
| 6 | 187.5 | 32.7 |
| 9 | 204.1 | 34.6 |
| **12** | **217.4** | **34.1** |

Generation settles by the third request — that is the experts paging back in
after the cache drop. Prefill is still climbing at the thirteenth. Part of that is
the 50-word vocabulary: 2,500 possible bigrams, so the bigram heads' working set
converges over the run in a way real text would not. The −23% is a point on a
curve whose end this run did not reach. A longer run on real prose is below.

### What whole-file `MADV_RANDOM` does (don't)

`llama-mmap.cpp` only applies `POSIX_MADV_RANDOM` under `if (numa)`, and
`ggml_is_numa()` is `n_nodes > 1` — false on this single-socket box regardless
of `--numa distribute`, which produced a byte-identical rerun. So the branch was
forced with a one-line patch, gated on an env var and otherwise inert:

```diff
-        if (numa) {
+        if (numa || getenv("LLAMA_MMAP_RANDOM")) {
```

Same 46G arm with it on:

| Request | Prefill | Generate |
|---|---|---|
| 0 | 3.4 | 8.3 |
| 3 | 17.4 | 4.5 |
| 6 | 32.8 | 18.8 |
| 9 | 128.0 | 31.9 |
| 12 | 185.9 | 34.1 |

80 GB of disk reads against 14 GB, and a five-times-slower warm-up. Killing
readahead for the engram also kills it for the experts — 32 GiB paging back in at
4 KiB per synchronous fault — and the experts need it far more than the engram
is hurt by it. The fix, if there is one, is per-tensor: `madvise(MADV_RANDOM)` on
the engram's byte range only, and since the hash runs host-side *before* the
gather, `MADV_WILLNEED` on all 16 × N rows at once so the kernel can overlap the
faults instead of taking them one at a time. Neither is tested here.

### Steady state on real prose

The synthetic vocabulary converges; real text does not. 30 requests per arm on
llama.cpp's own `docs/*.md`, chunked to ~1,300 characters (316–771 prompt
tokens, median ~365), 48 generated tokens each, same chunks in the same order
for both arms. Figures are the mean of requests 10–29; the first ten are
warm-up in the capped arm and flat in the other.

| | RAM (no cap) | SSD (46G cap) | Δ |
|---|---|---|---|
| Prefill | 203.7 tok/s | **134.3 tok/s** | **−34%** |
| Generate | 35.1 tok/s | **33.9 tok/s** | **−3.6%** |
| Wall-clock per request | 3.15 s | 4.2 s | +33% |
| Disk per request | 0 | ~850 MiB | — |
| File pages | 87.3 GiB | 37.0 GiB (32.3 needed) | experts resident |

Per-request prefill in the capped arm over requests 10–29 ranges 91–205 with no
upward trend, so this is the steady state, not another point on a warm-up
curve. The uncapped arm ranges 175–273 over the same requests — prefill rate
tracks prompt length in both — which is why the comparison is a ratio of means
over identical prompts rather than a pair of single numbers.

**Decode: −3.6%.** 16 rows a token, mostly cache-warm, the rest a 54 µs fault
each. This is the prediction from the drive's random-read latency and it held.

**Prefill: −34%.** A 365-token prompt gathers 5,840 rows before the first
layer's attention can run. Two costs stack: 5,840 synchronous faults at ~54 µs
is ~0.3 s, and 5,840 × 200 KiB of readahead is ~1.1 GB at a few GB/s, another
~0.3–0.4 s. Against 1.8 s of prefill that is the measured penalty, and it says
the two mechanisms contribute about equally — which matters for the fix,
because `MADV_RANDOM` alone only removes the second.

**Disk: ~2 MiB per token, for 1.4 KB of weights.** Still the readahead
multiplier. Real prose has a wider working set than 50 words, so the churn in
7 GiB of slack never settles.

So the answer to "can 29% of this model live on NVMe": **for decode, yes, at
4%; for prefill, at a third of the speed, with the current mmap path.** The
saving is 26.8 GiB of RAM. On a 186 GB box that buys nothing; on a 64 GB box it
is the difference between this model loading and not.

## Method: five ways this measurement lied before it worked

Each of these produced a clean-looking table with no signal in it.

1. **`curl -s …/health && ready` passes on HTTP 503.** curl exits 0 for any
   response. The check cleared while the model was still loading; every request
   got a 503; six configurations "passed" with empty timing columns. Readiness is
   a real generation returning 200, nothing less.
2. **`setsid systemd-run --scope` does not attach.** The process landed in the
   SSH session's scope with `memory.max = max`. Three caps, identical RSS. A
   transient `--unit` service attaches; and the cap is read back from
   `/sys/fs/cgroup/…/memory.max` before anything is measured.
3. **Page cache is charged to whoever faulted it first.** The 87 GiB were already
   cached from earlier runs, owned by an old cgroup; the new one held 1.4 GiB and
   the cap had nothing to bite. Dropping the cache before each arm — `fadvise`
   works once nothing maps the file — makes the new process the owner.
4. **The same prompt every time measures the cache, not the disk.** Zero disk
   reads at every cap looked like "engram on SSD is free". It was "the same
   trigrams are free the second time". Varied prompts fixed it.
5. **A cap that evicts the experts measures the experts.** 34G left 27 GiB of
   file pages against 32.3 GiB of experts and reported −38%. File pages against
   expert bytes is the check; it is in every table above.
6. **Thirteen requests from a 50-word list is a warm-up curve.** 2,500 possible
   bigrams converge into cache and the prefill number keeps rising until the run
   ends — the −23% was wherever the run happened to stop. Real prose, thirty
   requests, mean over the back twenty: −34%, and flat.

## Still to measure

- Context ceiling via ctxprobe, on two cards at a placement that leaves room.
- `-ncmoe 4 --tensor-split 26,22 -ub 2048`: the combination the prefill sweep
  points at, and whether it clears 1,527.
- `UD-Q3_K_XL` at `-ncmoe 0`: the 88 tok/s floor, and the ~1,700 tok/s prefill
  extrapolation.
