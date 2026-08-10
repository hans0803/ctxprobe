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
怎麼在你自己的卡上做這個選擇：[model-quant.zh-TW.md](../docs/model-quant.zh-TW.md)。

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
這個設定在做什麼、什麼時候不該用：[kv-cache-quant.zh-TW.md](../docs/kv-cache-quant.zh-TW.md)。

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

## Lazy 與 eager kernel 載入

CUDA 12 是在「某個 kernel 第一次被碰到」時才載入它的程式碼。
這使得天花板取決於「這一輪剛好用到了哪些 kernel」。
`CUDA_MODULE_LOADING=EAGER` 則是在啟動時就全部載入：

| Module loading | 最大 context | 差距 |
|---|---|---|
| `LAZY`（預設） | 34,816 | — |
| **`EAGER`** | **25,344** | **−9,472（−27%）** |

這個落差不是誤差。它代表 34,816 裡面有多少是押在
「這個工作負載永遠不會實例化另一個 kernel」上面。
換個 sampler、加個 grammar、餵進不同的 batch 形狀，
一個通過測試的 config 仍然可能在正式環境掛掉 ——
LAZY 的數字是**這次測試**的性質，EAGER 的數字才是**這個 config** 的性質。

兩者失敗的位置也不同，這反過來確認了機制：

| 模式 | 失敗在哪 |
|---|---|
| LAZY | `cudaFuncSetAttribute` @ `mmq.cuh:1375` —— 載入 kernel |
| EAGER | `alloc` @ `ggml-cuda.cu:589` —— CUDA 記憶體池 |

注意 EAGER **並不會**讓 prompt 階梯變得多餘。
25,600 在 EAGER 之下仍然是「短 prompt 過、長 prompt 死」，
因為 ggml 的記憶體池本來就會依需求成長，跟 kernel 何時載入無關。
在任何一種模式下，「開得起來」都不等於通過。

該用哪個數字：如果你能掌控工作負載、又想要最多的 context，用 `LAZY`；
如果這個 config 得撐住別人丟過來的任何東西，用 `EAGER`。

### 記憶體池對什麼有反應

kernel 都預先載入之後，剩下的失敗來源就是記憶體池 ——
而它的成長跟 batch 形狀有關，跟 prompt 長度無關。
在 EAGER 之下對 25,600 逐級加長 prompt：

| Prompt | `-fa auto` | `-fa on` | `-fa off` |
|---|---|---|---|
| 256 | ok | ok | *根本啟動不了* |
| **512** | **死亡** | **死亡** | — |

512 正是 `n_ubatch` 的預設值：第一個完整的 micro-batch，
池在這裡擴張到它的工作上限。更長的 prompt 會被切成 512 個 token 的
micro-batch，所以超過這一點之後形狀就不再改變。

`-fa auto` 和 `-fa on` 完全一致，而 `-fa off` 根本啟動不了：

```
llama_init_from_model: V cache quantization requires flash_attn
```

**量化的 V cache 會把 flash attention 釘在開啟狀態**，不管 `-fa` 寫什麼。
啟動 log 和 `/props` 都不會回報這件事，所以 ctxprobe 現在會印出來 ——
它會改變答案，而兩台機器對同一個模型得到不同結果時，
否則根本看不出原因。

這也讓 prompt 階梯的每一階從「湊整數」變成「有依據」：
64 是 `MMQ_DP4A_MAX_BATCH_SIZE`、512 是 `n_ubatch`、4096 跨過 `n_batch`（2048）。
目前量到的每一次失敗，都精確落在前兩者之一。

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

## 8GB 版本會是什麼樣子

5060 Ti 的 8GB 與 16GB 版本出自同一顆 GB206 die，核心數、時脈、
448 GB/s 頻寬都相同 —— 所以在這裡佔住 7899 MiB 就能重現 8GB 版本的預算。
這個替代為什麼成立、以及怎麼複現：[spill-cost.zh-TW.md](../docs/spill-cost.zh-TW.md)。

保留 7799 MiB，context 固定 4096：

| `-ngl` | 在 GPU 上的層數 | 結果 | Prefill | 生成 |
|---|---|---|---|---|
| 999 | 全部 65 層 | **LOAD_FAIL** | — | — |
| 32 | 32 | **LOAD_FAIL** | — | — |
| **28** | 65 層中的 28 層 | PASS | 391.21 tok/s | **5.11 tok/s** |
| 24 | 65 層中的 24 層 | PASS | 367.60 tok/s | 4.63 tok/s |

跟同一個模型完整放在 16GB 卡上對比：

| | 16GB —— 全部 65 層 | 8GB —— 65 層中的 28 層 |
|---|---|---|
| 生成 | 約 26 tok/s | **5.11 tok/s** |
| Prefill | 約 900 tok/s | 391 tok/s |
| 最大 context | 34,816 | 測試值 4,096 |

27B 的 IQ4_XS 在 8GB 卡上，以任何堪用的標準來說都裝不下。
上表的峰值顯存包含模擬程序佔住的 7899 MiB。

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
