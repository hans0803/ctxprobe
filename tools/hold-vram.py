#!/usr/bin/env python
"""Hold VRAM so the card behaves like a smaller one.

The 5060 Ti ships as 8GB and 16GB variants off the same GB206 die — same 4608
cores, same clocks, same 128-bit GDDR7 at 448 GB/s. Capacity is the only
difference, so squeezing a 16GB card down to an 8GB budget gives speed numbers
that transfer to the real 8GB part.

Usage: hold_vram.py TARGET_FREE_MIB
"""
import sys, time, torch

target = int(sys.argv[1])

# The first mem_get_info call creates this process's CUDA context (~138 MiB),
# so the free figure it returns already accounts for our own overhead.
free, total = torch.cuda.mem_get_info()
free_mib, total_mib = free // 2**20, total // 2**20
print(f"before: {free_mib} MiB free of {total_mib} MiB allocatable", flush=True)

need = free_mib - target
if need <= 0:
    print(f"nothing to do: already at or below {target} MiB", flush=True)
else:
    blocks, held, CHUNK = [], 0, 64
    while held < need:
        size = min(CHUNK, need - held)
        try:
            blocks.append(torch.empty(size * 2**20, dtype=torch.uint8, device="cuda"))
        except RuntimeError:
            break          # fragmentation floor — close enough
        held += size
    print(f"holding {held} MiB", flush=True)

free2, _ = torch.cuda.mem_get_info()
print(f"READY {free2 // 2**20}", flush=True)

while True:
    time.sleep(3600)
