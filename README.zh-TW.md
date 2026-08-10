# ctxprobe

[English](README.md) · **繁體中文**

**估算器告訴你這個 context「應該」裝得下，ctxprobe 證明它真的跑得動。**

找出一張顯卡上，GGUF 模型**實際**能運作的最大 context —— 方法是真的把它啟動、
用真實長度的 prompt 灌滿視窗、確認它沒有掛掉。一支 bash 腳本，除了
`llama-server` 和 `python3` 之外沒有其他相依。

```bash
# 不指定範圍時，會從 2048 一路搜尋到模型的訓練 context 上限；
# 縮小範圍只是為了少啟動幾次
./ctxprobe ~/models/Qwen3.6-27B-IQ4_XS.gguf --min 32768 --max 36864 -- --reasoning-budget 0
```

RTX 5060 Ti 16GB 的實際輸出：

```
  model     : Qwen3.6-27B-IQ4_XS.gguf (15G)
  gpu       : NVIDIA GeForce RTX 5060 Ti
  vram      : 16311 MiB reported by nvidia-smi, 15 MiB already in use
              15849 MiB actually allocatable (462 MiB is driver reserve)
  kv cache  : q8_0 | slots: 1 | step: 256 | long prompt fills 95%
  cuda      : CUDA_MODULE_LOADING=LAZY (default) | -ngl 999
  attention : flash_attn=on (forced by quantised V cache)

CONTEXT    RESULT      PEAK_VRAM   PREFILL    GENERATE   FILLED
------------------------------------------------------------------------
32768      PASS        15767       968.60     28.55      4042
36864      DECODE_OOM  15845       n/a        n/a        —
34816      PASS        15845       968.55     28.55      4042
35840      DECODE_OOM  15805       n/a        n/a        —
35328      DECODE_OOM  15787       n/a        n/a        —
35072      LONG_OOM    15847       94.03      26.47      died@64

Validating 34816 at 95% fill...
  confirmed: 34816 (32826 tokens filled)

Largest context that actually runs: 34816 tokens
  peak VRAM 15845 MiB | prefill 859.74 tok/s | generate 24.89 tok/s
```

不到兩分鐘。注意 `died@64`：35072 **載入成功、18-token 的 prompt 也能全速生成**，
卻死在一個 64-token 的 prompt 上 —— 這一行是任何載入期估算都產生不出來的。
接著勝出者會再用完整長度的 prompt 驗證一次，它的速度數字就來自那一輪。

## 為什麼需要這個

VRAM **估算器**已經有很好的了 —— [oobabooga 的迴歸公式](https://oobabooga.github.io/blog/posts/gguf-vram-formula/)
（19,517 筆實測）、llama.cpp 自己的 `llama-fit-params`，還有半打計算器。
它們回答的是**「裝得下嗎？」**，而且回答得很好。

但問題有三層，不是兩層：

| | 回答的問題 | 這個模型、這張卡 |
|---|---|---|
| 估算器 | 載得進去嗎？ | 約 35K |
| `ctxprobe` | 跑得起**我剛剛跑的東西**嗎？ | **34,816** |
| `ctxprobe --eager` | 跑得起**任何東西**嗎？ | **25,344** |

後兩者相差 **9,472 個 token —— 27%**。
那個落差就是天花板裡「押在『這個工作負載永遠不會碰到還沒碰過的 kernel』」上的部分。
CUDA 12 是在第一次被碰到時才載入 kernel 程式碼，
所以一個通過所有測試的 config，仍然可能之後因為換了 sampler、
加了 grammar、或換個 batch 形狀而掛掉。
`--eager` 會把所有 kernel 提前載入，回報的是「不管怎樣都成立」的數字。

沒有人量中間那一列，而幾乎沒有人知道最後一列的存在。

RTX 5060 Ti（16 GB）、Qwen3.6-27B-IQ4_XS + q8_0 KV 的實測：

| Context | 載入 | 短 prompt | 30K token 的 prompt |
|---|---|---|---|
| 34816 | 成功 | 25.99 tok/s | **正常運作** |
| 35072 | 成功 | 25.98 tok/s | **CUDA OOM，服務崩潰** |

35072 載入完全正常、生成速度也是滿的。然後一個真實長度的 prompt 就殺了它：

```
launch_mul_mat_q at mmq.cuh:1375
cudaFuncSetAttribute(mul_mat_q<type, J, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)
CUDA error: out of memory
```

**長 prompt 會用到短 prompt 從來碰不到的 CUDA kernel。**
llama.cpp 的量化矩陣乘法是以 batch 形狀作為模板參數的，
所以較大的 prefill 會實例化另一個 `mul_mat_q` variant ——
而在 CUDA 的 lazy module loading（CUDA 12 之後的預設）之下，
第一次觸碰某個 kernel 才會把它的程式碼載入 device memory。
顯存見底時，失敗的就是這個載入。

注意這**不是**「推論過程中配置量成長」。
以 50 ms 為間隔對完整的 prefill 與 decode 取樣，顯存從頭到尾都停在 15845 MiB。
它只是需要記憶體去載入一個還沒載入過的 kernel。

有兩個實驗把這件事釘死。強制所有 kernel 在啟動時就載入，
會把神秘的執行期崩潰變成確定性的載入期失敗：

```
$ CUDA_MODULE_LOADING=EAGER llama-server ... -c 35072
allocating 251.53 MiB on device 0: cudaMalloc failed: out of memory
llama_init_from_model: failed to allocate compute pp buffers
```

而致命的 prompt 遠比「灌滿視窗」短得多。
對兩個 config 逐級加長 prompt：

| Prompt | 34,816（通過） | 35,072（失敗） |
|---|---|---|
| 16 tokens | ok | ok |
| 32 tokens | ok | ok |
| **64 tokens** | ok | **死亡** |
| 2,025 tokens | ok | — |

**64 個 token 就足以區分兩者。** ctxprobe 自己的暖身 prompt 是 18 tokens，
這正是它為什麼非得靠長 prompt 才發現問題 ——
不是因為「灌滿視窗」有什麼特別，而是因為 18 tokens 落在
「會實例化新 kernel variant」的門檻以下。
prompt 階梯就是利用這一點：依序爬 64 → 512 → 4096 → 完整長度，
第一次死亡就停 —— 註定失敗的 config 幾秒內就被淘汰，
而不是等一次 30K token 的 prefill 跑完。

因此 `PASS` 的定義是：撐過一個**灌滿視窗 95%** 的 prompt，而且真的吐得出 token。
這個百分比是量出來的、不是估的 —— 長度透過伺服器自己的
`/v1/chat/completions/input_tokens` 端點迭代逼近，所以連 chat template 的包裝都算進去了，
而那層包裝正好就是把「接近滿」的 prompt 推過界的元兇。
`FILLED` 欄位回報的就是實際灌進去的 prompt 大小。

只要求回傳 8 個 token。要證明的是「在那個深度下 prefill 撐得住、模型還開得了口」，
而不是它寫得多快。

## 從這裡開始

單卡上，有兩個設定決定了你大部分的結果：

- **[你該下載哪一個量化？](docs/model-quant.zh-TW.md)** ——
  看懂 `Q4_K_M` / `IQ4_XS` 這些名字，以及最重要的那條規則：
  「**完整**裝得下的最大量化」勝過「會溢出的更好量化」。
- **[KV cache：為什麼用 q8_0](docs/kv-cache-quant.zh-TW.md)** ——
  長 context 上槓桿最大的設定。在這裡，一個參數換到多 72% 的 context。
- **[裝不下的時候，代價是什麼](docs/spill-cost.zh-TW.md)** —— 實測：
  溢出 57% 的層會慢五倍，以及怎麼量一張你沒有的顯卡。

接著看 [gotchas.zh-TW.md](docs/gotchas.zh-TW.md) 了解什麼會悄悄吃掉顯存，
以及 [results/](results/) 裡的實測配置。

## 三件會吃掉你顯存的事

都是踩過坑才知道的；三件事估算器全都看不到。

**1. `nvidia-smi` 高估了你的顯存。** 在一張 16 GB 的卡上，有 462 MiB 是驅動保留區，
任何配置都永遠拿不到：

```
nvidia-smi 報的 total      16311 MiB
CUDA 實際能配置的          15849 MiB   <- 真正的天花板
```

要用 `torch.cuda.mem_get_info()[1]` 算餘裕，不是 `nvidia-smi`。ctxprobe 兩個都印。

**2. `llama-server` 預設開 4 個 slot。** `-c 8192` 實際上會替
**4 × 8192 = 32768** 個 token 配置 KV。在我們的測試卡上，光是加 `--parallel 1`
就讓可用 context 變成四倍（4096 → 16384），完全還沒調其他東西。
ctxprobe 預設就是 `--parallel 1`。

**3. `n_ctx` 會被對齊到 256 的倍數。** 出自 `llama-context.cpp`：

```c
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

所以 `-c 35000` 會被靜靜變成 35072。用 1024 當步進（很自然的習慣）每一階會跳過
三個可測的點；ctxprobe 預設步進是 256。

另外三個坑收在 [gotchas.zh-TW.md](docs/gotchas.zh-TW.md)：
thinking 模型回傳空輸出、桌面程序佔住顯卡、子程序已死卻仍被回報為健康。

## 用法

```
ctxprobe MODEL.gguf [選項] [-- 額外的 llama-server 參數]

  --min N        搜尋下限（預設 2048）
  --max N        搜尋上限（預設：模型的訓練 context，上限 131072）
  --step N       粒度（預設 256）
  --kv TYPE      KV cache 型別：q8_0（預設）、f16、q4_0
  --parallel N   server slot 數（預設 1）
  --ngl N        放上 GPU 的層數（預設 999 = 全部；調低會溢出到系統記憶體）
  --list "A B C" 直接測這些數值，不做二分搜尋
  --fill PCT     長 prompt 要灌多滿（預設 95，由伺服器 tokenizer 實測）
  --json         機器可讀的輸出
  --keep         保留每一輪的 log
```

預設用二分搜尋，所以找出上限只需要約 log₂(範圍) 次啟動，而不是每個候選值各跑一次。
每次啟動都要重新載入模型，所以請預期會花上幾分鐘。

額外的 `llama-server` 參數放在 `--` 後面直接傳遞：

```bash
# Qwen3.6 這類 thinking 模型會把整個 token 預算燒在推理段上，
# 不關掉的話 content 會是空的
./ctxprobe model.gguf -- --reasoning-budget 0
```

`LLAMA_SERVER=/path/to/llama-server` 可覆寫 binary 的搜尋結果。
`CUDA_VISIBLE_DEVICES` 用來選卡（設計上只處理單卡）。

要顯示真實的可配置總量，需要機器上某個 Python 能 import `torch`
（會自動搜尋 conda envs；`CTXPROBE_PYTHON` 可以指定特定的一個）。
必須是 `cudaMemGetInfo` —— NVML 和 `nvidia-smi` 回報的是板子的物理容量，
那個數字正好就是把保留區藏起來的元兇。沒有 torch 一切照常運作，只是少那一行。

## 實測資料

| GPU | 模型 | 量化 | KV | 最大 context | Prefill | 生成 |
|---|---|---|---|---|---|---|
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | q8_0 | 34,816 | 858 tok/s | 24.6 tok/s |
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | f16 | 20,224 | 923 tok/s | 26.9 tok/s |
| RTX 5060 Ti 8GB *(模擬)* | Qwen3.6-27B | IQ4_XS | q8_0 | 4,096 † | 391 tok/s | 5.11 tok/s |

† 裝不下：65 層中只有 28 層在 GPU 上，其餘在系統記憶體。
見 [spill-cost.zh-TW.md](docs/spill-cost.zh-TW.md)。

前兩行是同一個模型、同一張卡，只差一個參數。完整的實測過程
（包含騰出顯存後上限怎麼往上跳）在
[rtx5060ti-16gb-qwen3.6-27b.zh-TW.md](results/rtx5060ti-16gb-qwen3.6-27b.zh-TW.md)。

速度來自「灌滿視窗的 prompt」，所以描述的是滿載視窗而不是空視窗的狀態。
不要把 f16 那一行讀成「f16 生成比較快」—— 它跑在 20K context 而另一行是 34K，
生成速度本來就會隨視窗填滿而變慢。

歡迎補上其他顯卡的資料；`--json` 的輸出就是設計來直接貼進去的。

## 範圍

**一張卡，一個數字：它的顯存到底撐得住多長的 context。**

`PASS` 代表整個模型都跑在 GPU 上（`-ngl 999`，沒有 CPU offload），
所以系統記憶體不進入預算，這個上限可以完全歸因於這張卡。

溢出到顯卡之外並不是禁區，只是不算通過。量出「溢出的代價」正是說明
「為什麼要留在顯存裡」最清楚的方式，所以那些測試也屬於這裡 ——
只是會標示為對照組，而不是當成一個可用的配置。

多卡配置不在範圍內；`llama-fit-params` 已經做得很好。

**PASS 沒有涵蓋到什麼。** 階梯只要求回傳 8 個 token。
真實使用是在滿的 KV cache 上生成好幾千個 token ——
batch 為 1、走的是另一條矩陣乘法路徑（MMVQ 而不是 MMQ）、而且持續好幾分鐘。
那條路徑在這裡從來沒有被走過。
就機制而言沒有理由認為它會配置得比 prefill 更多，
但「理論上不會」不等於「量過了」，
而這個工具存在的理由，正是這兩者的差別。

## 授權

MIT
