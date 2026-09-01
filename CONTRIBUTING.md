# Contributing

**English** · [繁體中文](CONTRIBUTING.zh-TW.md)

The useful thing to send is **a measurement from a card we don't have**. Every
number in [results/](results/) comes from two machines owned by one person,
which is exactly as narrow as it sounds.

## Sending a card

```bash
./ctxprobe ~/models/YourModel-Q4_K_M.gguf --json > result.json
```

Open an issue with that file pasted in, plus:

- **the exact model file** — a Hugging Face repo id and filename, not "Qwen 27B
  Q4". Two files with the same quant name can differ by half a bit per weight;
  see [model-quant.md](docs/model-quant.md).
- **anything else on the card** — a desktop session, another process. The run
  prints what it found, but it cannot know what will show up later.
- **the llama.cpp commit.** Ceilings move between versions.

That is enough. You do not need to write a page.

### What makes a result hard to use

- **`validated_at_fill: false`.** The ladder cleared the size but a full window
  did not, and the search could not find one below it that held. The number is
  not a ceiling; the log is the interesting artefact. Re-run with `--keep` and
  attach it.
- **`--ngl` below the layer count.** Then part of the model is in system RAM
  and the ceiling is not a property of the card. Deliberate spill measurements
  are welcome — say so, and see [spill-cost.md](docs/spill-cost.md).
- **`--parallel` above 1** without saying so. KV scales with it, so the number
  means something different.

## The JSON

`--json` writes one object to stdout; progress goes to stderr. Schema 3:

| Field | Type | Meaning |
|---|---|---|
| `schema` | int | Bumped when the shape changes. Currently **3**. |
| `model` | string | Basename of the GGUF passed in. |
| `gpu` | string | As `nvidia-smi` names it. |
| `max_context` | int | **The answer.** Largest context that ran. |
| `validated_at_fill` | bool | Whether `max_context` survived a prompt filling `fill_percent` of the window. **If false, `max_context` is unconfirmed** — the ladder cleared it and a full window did not. |
| `ladder_max_context` | int | What the 64/512/4096 ladder alone would have reported. Equal to `max_context` unless a full window failed and the search bisected down — see [gotchas.md](docs/gotchas.md) #11 for the one architecture where the gap is large. |
| `peak_vram_mib` | int\|null | Highest sampled usage on the probed device during the winning run. Not a headroom estimate — [gotchas.md](docs/gotchas.md) #5. |
| `prefill_tps` | float\|null | From the fill validation, so against a loaded window. |
| `generate_tps` | float\|null | Same run. Decode into a nearly-full cache, not an empty one. |
| `kv_cache_type` | string | `--kv`. |
| `slots` | int | `--parallel`. KV is allocated per slot. |
| `n_gpu_layers` | int | `--ngl`. Below the model's layer count means it spilled. |
| `fill_percent` | int | `--fill`, default 95. |
| `cuda_module_loading` | string | `LAZY (default)` or `EAGER`. The two give different ceilings; see the README. |
| `flash_attn` | string | What the server actually used — a quantised V cache pins it on regardless of `-fa`. |
| `vram_total_mib` | int | What `nvidia-smi` reports. |
| `vram_allocatable_mib` | int\|null | What `cudaMemGetInfo` reports, which is smaller. Null without an importable `torch`. |
| `vram_held_by_others_mib` | int | Device memory in use before the run started. |

Per-boot rows go to `results.tsv` in `--out`, one line each:

```
context  result  peak_vram_mib  prefill_tps  generate_tps  prompt_tokens
```

Rows prefixed `final:` in the `result` column are the fill validations. The
rest are ladder probes, and their speeds come from whichever rung the run
reached — so they describe a lightly-loaded window and are not comparable to
the `final:` figures.

`result` is one of `PASS`, `LOAD_FAIL`, `DECODE_OOM`, `PREFILL_OOM`, `SILENT`.
`prompt_tokens` carries `died@N` when a rung killed the run, naming the rung.

## Changing the script

It is one bash file, `set -uo pipefail`, no dependencies beyond `llama-server`
and `python3`. CI runs `shellcheck -S warning`, checks `--help` does not crash,
and checks that four kinds of bad arguments are rejected rather than crashing.

Concurrent runs on one machine need different `--port` values. The script
takes an atomic lock per port, because the startup port check is a moment in
time and two runs can both pass it — after which the loser's server fails to
bind while its requests reach the winner's, and it reports a PASS for a
configuration that never ran.

Two more things worth knowing before editing:

- **`-e` is deliberately off.** Probes are expected to fail; that is the
  measurement. Check exit codes where you mean to.
- **Report the result, not the call.** A `curl` that returns 0 has not told you
  the server worked — it returns 0 for HTTP 503, and it returns a stale body if
  the connection dropped mid-request. Readiness in this script is a real
  generation returning 200. Several sections of
  [rtx5090-32gb-qwen3.8-flash-next.md](results/rtx5090-32gb-qwen3.8-flash-next.md)
  exist because that rule was broken.

## Adding a results page

Only if you want to. The bar is that **every number is reproducible from what
is on the page** — the exact flags, the exact file, the machine. Where a number
comes from a mechanism rather than a measurement, say which. Where a later run
contradicts an earlier page, the earlier page gets a correction rather than a
quiet edit; `results/rtx5060ti-16gb-qwen3.6-27b.md` has an example.

Pages are bilingual (`*.md` and `*.zh-TW.md`) and the numbers must match
between them. English-only is fine to send; say so and it can be translated.
