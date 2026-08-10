# RTX 5060 Ti 16GB — Qwen3.6-27B-IQ4_XS

[English](rtx5060ti-16gb-qwen3.6-27b.md) · **繁體中文**

完整實測，2026-08-10。這份資料就是這個工具的來源。
涵蓋 `q8_0` 與 `f16` 兩種 KV cache，讓「不量化的代價」是看得到的，而不是用猜的。

六次啟動即可復現這個上限：

```bash
ctxprobe Qwen3.6-27B-IQ4_XS.gguf --min 32768 --max 36864 -- --reasoning-budget 0
```

## 硬體

這裡的一切都由顯卡決定。模型完全跑在顯存裡（`-ngl 999`，沒有 CPU offload），
所以系統記憶體從來不進入預算。

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 回報 16311 MiB，可配置 15849 MiB** |
| 驅動 | 595.84 |
| llama.cpp | `0a50d99`，CUDA build |
| 主機 | i7-14700、64 GB DDR4 —— 對以下任何數字都不構成因素 |

## 模型

[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) — `Qwen3.6-27B-IQ4_XS.gguf`，15.44 GB。

Qwen3.6-27B 有 64 層，但其中**只有 16 層是 full attention**（`full_attention_interval: 4`）；
其餘 48 層是 linear attention，state 大小固定。
因此 KV cache 比一般的 27B 便宜得多 —— q8_0 之下約 **每 1K token 34 MiB**。

為什麼選 IQ4_XS：它是裝得下的最大 Q4 級量化。`Q4_K_S` 是 15.86 GB、
`Q4_K_M` 是 16.82 GB —— 在還沒有任何 KV cache 之前就已經超出這張卡。

## Context 上限

`--parallel 1`、`--cache-type-k/v q8_0`、`-ngl 999`、`--reasoning-budget 0`。

| Context | 結果 | 峰值顯存 | 生成 |
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

這裡刻意沒有 prefill 欄位。這些測試是用短 prompt 認證的，
而短 prompt 在這張卡上會回報約 98 tok/s 的 prefill，真實負載下卻是約 900 tok/s ——
那是固定開銷造成的假象，不是一個速率。prefill 只有在對照 prompt 長度時才有意義，
也就是下一節的內容。

**34,816 是上限**，並且在視窗灌到 95%（經伺服器 tokenizer 實測為 32,997 tokens）
的條件下再次確認。35,072 能載入、短 prompt 能全速生成，
然後在真實長度的 prompt 下 CUDA OOM，留下一個殭屍程序。
這是「必須用真實長度的 prompt 驗證」最清楚的一個論據。

生成速度在整個範圍內都平穩維持在約 26 tok/s ——
在撞牆之前，context 消耗的是記憶體，不是吞吐量。

## KV cache：q8_0 vs f16

同一個模型、同一張卡，只改 `--cache-type-k/v`。
兩個上限都是用二分搜尋、視窗灌到 95% 找出來的。

| KV 型別 | 最大 context | 峰值顯存 | Prefill | 生成 |
|---|---|---|---|---|
| `f16`（預設） | 20,224 | 15843 MiB | 922.54 tok/s | 26.94 tok/s |
| **`q8_0`** | **34,816** | 15845 MiB | 858.39 tok/s | 24.62 tok/s |

**量化 KV cache 換到了多 72% 的 context**（多出 14,592 個 token）。

直覺會猜「cache 減半、context 就該加倍」，但並沒有。
權重是固定的 15.44 GB，cache 完全碰不到那塊 —— 只有剩下的約 400 MiB 才是
cache 的預算，所以把每個 token 的成本減半，延長的是那塊剩餘量，而不是整個視窗。
模型相對於顯卡越大，這個倍率就越小。

完整的 f16 搜尋過程：

| Context | 結果 | 峰值顯存 | 實際灌入 |
|---|---|---|---|
| 8,192 | PASS | 15081 MiB | 7,733 |
| 16,384 | PASS | 15601 MiB | 15,424 |
| 18,432 | PASS | 15731 MiB | 17,287 |
| 19,456 | PASS | 15795 MiB | 18,324 |
| 19,968 | PASS | 15827 MiB | 18,897 |
| **20,224** | **PASS** | 15843 MiB | 19,203 |
| 20,480 | DECODE_OOM | 15783 MiB | — |
| 24,576 | LOAD_FAIL | — | — |

不要把生成那一欄讀成「f16 比較快」。那些測試跑在 20K context，
而 q8_0 是 34K，生成速度本來就會隨視窗填滿而變慢 ——
在同樣的 16,384 之下，f16 那輪是 27.28 tok/s，
跟 q8_0 在相近深度的數字差在雜訊範圍內。
KV 型別是記憶體決策，不是速度決策。

## Prefill 與 prompt 長度的關係

對照已部署的 34,816 配置量測。每個 prompt 都是隨機產生的，
避免伺服器的 prompt cache 影響結果。

| Prompt token 數 | Prefill | TTFT | 生成 |
|---|---|---|---|
| 799 | 836 tok/s | 0.96 s | 27.52 tok/s |
| 3,230 | 970 tok/s | 3.3 s | 26.67 tok/s |
| 12,649 | 945 tok/s | 13.4 s | 25.56 tok/s |
| 21,577 | 905 tok/s | 23.8 s | 24.94 tok/s |
| 25,424 | 890 tok/s | 28.6 s | 23.99 tok/s |
| 32,508 | 860 tok/s | 37.8 s | 23.50 tok/s |

Prefill 穩定在約 860–970 tok/s，衰減很輕微。TTFT 則隨長度線性成長，
是長 context 下的主要成本 —— 32K 時要等 38 秒才吐出第一個字。
生成速度從空視窗到滿視窗下降約 13%。

短 prompt 的每 token 速度反而**比較慢**（799 tokens 時只有 836 tok/s），
因為固定開銷還沒被攤提掉。

## 騰出顯存換到了什麼

在完全沒有更動模型的情況下，context 上限往上跳了兩次：

| 變更 | dGPU 閒置佔用 | 上限 |
|---|---|---|
| 預設（`--parallel 4`）、桌面接在 dGPU | 605 MiB | 4,096 |
| `--parallel 1` | 605 MiB | 16,384 |
| 顯示輸出改接內顯 | 162 MiB | 30,720 |
| GNOME Remote Desktop 改用內顯 | **15 MiB** | **34,816** |

光是 `--parallel 1` 就是 4 倍的收穫。把顯示輸出移到內顯
（BIOS：`Primary Display = IGFX`、`iGPU Multi-Monitor = Enabled`）釋放了 443 MiB。

GNOME Remote Desktop 在顯示輸出搬走之後**仍然**佔著 dGPU 的 130 MiB，
因為它為了 NVENC 載入 `libcuda` + `libnvidia-encode`。把它釘到 Mesa/Intel 才放掉：

```ini
# ~/.config/systemd/user/gnome-remote-desktop.service.d/igpu.conf
[Service]
Environment=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
Environment=__GLX_VENDOR_LIBRARY_NAME=mesa
```

事後驗證：該程序只載入 `libEGL_mesa` + `libgallium`，
`libcuda` 已不再出現在它的記憶體映射中。

## 推論過程中的顯存行為

以 50 ms 為間隔，對完整的 prefill + decode 週期取樣 `nvidia-smi`，共 481 個樣本：

```
min 15845 MiB, max 15845 MiB
```

**llama.cpp 的顯存用量是完全靜態的。** 所有 buffer 都在載入時配置完畢，
推論過程中不會成長。所以一個能載入、又能通過滿視窗 prompt 的配置，
不會單純因為推論本身而在之後 OOM —— 那點餘裕只需要撐住「其他程序來動這張卡」。

## 已排除：NVFP4

這個模型的兩個 NVFP4 版本對 16 GB 來說都太大了，
因為「NVFP4」是混合精度，而不是全部都 4-bit：

| 版本 | 大小 | 說明 |
|---|---|---|
| [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) | 21.94 GB | attention 與 `linear_attn` 各層是 FP8 |
| [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 23.44 GB | 同上，另外視覺塔維持 BF16 |
| GGUF IQ4_XS | **15.44 GB** | 幾乎所有權重都是約 4.25 bpw |

兩個版本裡真正是 4-bit 的只有 MLP 張量。硬體本身是支援的 ——
5060 Ti 是 Blackwell sm_120，有原生 FP4 —— 但 vLLM 無法把權重 offload 到 CPU，
所以「太大」就等於「根本啟動不了」。
