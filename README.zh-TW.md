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

六次啟動就夾出上限。注意最後一行：35072 **載入成功、短 prompt 也能全速生成**，
卻在真實長度的 prompt 下死掉 —— 這一行是任何載入期估算都產生不出來的。

## 為什麼需要這個

VRAM **估算器**已經有很好的了 —— [oobabooga 的迴歸公式](https://oobabooga.github.io/blog/posts/gguf-vram-formula/)
（19,517 筆實測）、llama.cpp 自己的 `llama-fit-params`，還有半打計算器。
它們回答的是**「裝得下嗎？」**，而且回答得很好。

但那跟**「真的能用嗎？」**不是同一個問題。

RTX 5060 Ti（16 GB）、Qwen3.6-27B-IQ4_XS + q8_0 KV 的實測：

| Context | 載入 | 短 prompt | 30K token 的 prompt |
|---|---|---|---|
| 34816 | 成功 | 25.99 tok/s | **正常運作** |
| 35072 | 成功 | 25.98 tok/s | **CUDA OOM，服務崩潰** |

35072 載入完全正常、生成速度也是滿的。然後一個真實長度的 prompt 就殺了它。
**prefill 的 compute buffer 會隨 prompt 長度成長**，而載入期的估算模型不包含這一項 ——
所以任何只用短 prompt 驗證的方法，都會回報假的通過。

因此 `PASS` 的定義是：撐過一個**灌滿視窗 95%** 的 prompt，而且真的吐得出 token。
這個百分比是量出來的、不是估的 —— 長度透過伺服器自己的
`/v1/chat/completions/input_tokens` 端點迭代逼近，所以連 chat template 的包裝都算進去了，
而那層包裝正好就是把「接近滿」的 prompt 推過界的元兇。
`FILLED` 欄位回報的就是實際灌進去的 prompt 大小。

只要求回傳 8 個 token。要證明的是「在那個深度下 prefill 撐得住、模型還開得了口」，
而不是它寫得多快。

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

另外三個坑 —— thinking 模型回傳空輸出、桌面程序佔住顯卡、
子程序已死但仍被回報為健康 —— 收在 [docs/gotchas.zh-TW.md](docs/gotchas.zh-TW.md)。

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

已測配置放在 [results/](results/)。歡迎補上其他顯卡的資料 ——
`--json` 的輸出就是設計來直接貼進去的。

| GPU | 模型 | 量化 | KV | 最大 context | Prefill | 生成 |
|---|---|---|---|---|---|---|
| RTX 5060 Ti 16GB | Qwen3.6-27B | IQ4_XS | q8_0 | 34,816 | 858 tok/s | 24.6 tok/s |

速度都是在「灌滿視窗的 prompt」之下量的。視窗空的時候生成會更快
（這張卡約 26 tok/s），並隨 context 填滿而衰減 —— 標示滿載時的數字比較誠實。

## 範圍

**一張卡，一個數字：它的顯存到底撐得住多長的 context。**

`PASS` 代表整個模型都跑在 GPU 上（`-ngl 999`，沒有 CPU offload），
所以系統記憶體不進入預算，這個上限可以完全歸因於這張卡。

溢出到顯卡之外並不是禁區，只是不算通過。量出「溢出的代價」正是說明
「為什麼要留在顯存裡」最清楚的方式，所以那些測試也屬於這裡 ——
只是會標示為對照組，而不是當成一個可用的配置。

多卡配置不在範圍內；`llama-fit-params` 已經做得很好。

## 授權

MIT
