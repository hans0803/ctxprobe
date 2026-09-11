# RTX 5090 32GB — Qwen3.8-Flash-Next（UD-IQ4_XS）

[English](rtx5090-32gb-qwen3.8-flash-next.md) · **繁體中文**

實測於 2026-08-27。一個 176.9 B 參數、磁碟上 87 GiB 的模型，跑在一張 32 GB 的卡上。
「裝得下」不是新聞 —— [DeepSeek-V4-Flash](rtx5090-32gb-deepseek-v4-flash.zh-TW.md)
早就示範過 137 GB 的模型這樣跑。新聞是：**這個模型有 29% 是一張 hash table，
每個 token 只碰 1.4 KB 的權重**，以及當你試著把它移出 RAM 預算時會發生什麼。

> ⚠️ **這次跑的是尚未合併的程式碼。** `qwen4exp` 的支援不在 llama.cpp master，
> 而在 [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)，
> 這裡 build 的是 `213df58`。後面有一個實驗對 `llama-mmap.cpp` 加了一行
> 以環境變數控制的補丁，預設關閉，用到的地方會標出來。
>
> 除非該節另有說明，速度數字都是在預設的 `n_ubatch` 512 和 f16 KV 之下。
> Context 天花板那一節講的與其說是那些數字，不如說是階梯漏掉了什麼 ——
> 以及因此對 ctxprobe 做的修改。

## 硬體

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5090 — 32607 MiB，透過 `CUDA_VISIBLE_DEVICES=1` 用單卡** |
| CPU | AMD Ryzen 9 9950X，16 核 / 32 執行緒，單一 NUMA node |
| 記憶體 | 186 GB DDR5，雙通道 |
| NVMe | Kingston SFYRD2000G（PCIe 4.0），ext4，`read_ahead_kb` 128 |
| llama.cpp | PR #27742 @ `213df58`，build 861，CUDA 12.8（`sm_120`） |
| 模型 | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ4_XS`，87.24 GiB，3 個分片 |

第二張 5090 全程在服務一個不相干的 vLLM 實例；這裡每個數字都是單卡的。

在相信這個 build 之前，先在兩個 backend 上跑了 `test-llama-archs -a qwen4exp`：
**CUDA OK（NMSE 8.51e-08）、CPU OK（0.00e+00）**，跟 PR 自己宣稱的一致。
`Meta` backend 會 assert（`ggml-backend-meta.cpp:756`）—— 那是這個分支裡真實的 bug，
但落在明確指定 `-ngl` 時會跳過的圖分析路徑上。

## 架構

```
qwen4exp.block_count                = 48
qwen4exp.expert_count               = 512
qwen4exp.expert_used_count          = 10       <- 512 選 10
qwen4exp.expert_feed_forward_length = 640
qwen4exp.embedding_length           = 2560
qwen4exp.full_attention_interval    = 4        <- 12 層 full，36 層 gated delta-net
qwen4exp.attention.indexer.top_k    = 2048     <- QSA 稀疏注意力
qwen4exp.ple.layers                 = [1]
qwen4exp.context_length             = 262144
```

位元組落在哪，用 ggml 各型別的 block 大小算出來的，不是看標籤：

| | 參數 | GiB | bpw |
|---|---|---|---|
| **MoE experts** | 120,795,955,200 | **55.43** | 3.942 |
| **Engram（n-gram 表）** | 51,200,245,760 | **26.82** | 4.500 |
| 其他 | 3,669,736,320 | 3.86 | 9.042 |
| embed / output | 1,277,962,240 | 1.12 | 7.536 |
| **總計** | 176,943,899,520 | **87.24** | 4.235 |

## Engram：模型的 29%，每 token 1.4 KB

`per_layer_token_embd.weight` 是**一個** tensor，形狀 **(160, 320,001,536)**、型別
`IQ4_NL` —— 51.2 B 參數、26.82 GiB —— 而且**只存在於第 1 層**（`ple.layers = [1]`）。
「Per-layer embeddings」講的是它注入的位置，不是有幾份。

3.2 億列分成 16 個 head 區段，每段約 20,000,0xx 列；
16 = `(ngram_size 3 − 1) × heads_per_ngram 8`：8 個 bigram head 加 8 個 trigram head。
列的選擇是 host 端算的 hash，因為 ggml 既沒有 int64 也沒有 xor：

```
mixed_n = (t[p]·m0) ⊕ (t[p−1]·m1) ⊕ … ⊕ (t[p−n+1]·m_{n−1})     n = 2, 3
row_h   = mixed_n mod vocab[h] + offset[h]
```

乘數就寫在檔案裡（`ple.layer_multipliers = [23703573157769, 20109073645365,
8052911324071]`）；EOS 會重置視窗。

每個 token 取 16 列、每列 160 個值 —— **1,440 bytes 的權重、16 個 page、
每次 forward 只做一次** —— 然後過一個 key/value 投影和一個對殘差流做內積的 sigmoid 閘，
決定注入多少。對照 expert 每 token 1.16 GB，差 80 萬倍，
這就是為什麼這張表值得跟檔案的其餘部分分開處理。

它不能移植。這些列是在特定的 hash 碰撞模式下訓練出來的
（trigram 空間 248,320³ 塞進每 head 2,000 萬列，每列平均被 7.6 × 10⁸ 個 trigram 共用）；
換 tokenizer、換乘數、換 vocab 大小，整張表就變成雜訊。

## Expert 的擺放位置決定生成速度

`-ncmoe N` 把前 N 層的 expert 留在 CPU。`-c 8192`、`--threads 16`、
生成 128 token、warm：

| CPU 上的層 | GPU 上 | 峰值顯存 | 生成 | ms/token |
|---|---|---|---|---|
| 48 | 0 | 6,487 | 24.98 | 40.03 |
| 40 | 8 | 16,387 | 28.89 | 34.61 |
| 34 | 14 | 23,215 | 31.88 | 31.37 |
| 30 | 18 | 28,161 | 34.40 | 29.07 |
| **28** | **20** | **30,435** | **35.53** | **28.15** |
| 26 | 22 | — | — | **載入失敗** |

**20 層是這張卡的天花板。** 每層 expert 約吃 1,192 MiB 顯存（從差值算的；
55.43 GiB / 48 預測 1,182），所以 22 層要 32,711 MiB，卡只有 32,607。

對五個通過的點做最小平方：

```
t_token = 11.32 ms + 28.44 ms × (CPU 上的層數 / 48)

殘差全部在 0.7 ms 以內
```

**28.44 ms 是 DDR5 讀 expert 的時間**，而且已經接近最佳。每 token CPU 要讀
48 × 10 × 4,915,200 個參數、3.942 bpw = **1.1625 GB**：

```
1.1625 GB / 28.44 ms = 40.9 GB/s
```

對照這台實測能拉到的：

| 執行緒 | 循序讀 |
|---|---|
| 1 | **50.3 GB/s** |
| 4 | 46.2 |
| 8 | 45.7 |
| 16 | 44.5 |
| 32 | 43.8 |

**40.9 對 44.5 是 92%。** Expert 這條路是頻寬受限的。單核就幾乎打滿雙通道 DDR5，
所以執行緒加上去沒用、反而有害：

| 執行緒 | 生成（`-ncmoe 48`） |
|---|---|
| **16** | **25.10** |
| 24 | 23.90 |
| 28 | 23.72 |

**11.32 ms 是地板** —— GPU 上的圖、attention、delta-net、QSA indexer、PLE 閘 ——
再多顯存也消不掉。它把這個模型在這個 build 上的上限釘在 **88.3 tok/s**，
而那要 48 層 expert 全部常駐：6,487 + 48 × 1,192 = 63.7 GB。兩張 5090，不是一張。

## 卡不是瓶頸

`-ncmoe 48` 之下，生成中取樣：

| | |
|---|---|
| GPU 使用率 | **19–20%** |
| GPU 記憶體控制器 | 7–8% |
| GPU 功耗 | 113 W（滿載約 575 W） |
| CPU | **32 執行緒的 50.1% user** —— 16 條滿載 |
| 顯存 | 6,487 MiB / 32,607 |

**一個 176.9 B 的模型跑在 6.5 GB 顯存上，卡有 80% 在閒置。**
把 expert 層搬上去，就是把閒置的矽換成 42% 的吞吐。

## Prefill：`-ub`，以及顯存得從哪裡來

機制跟 [DeepSeek-V4-Flash](rtx5090-32gb-deepseek-v4-flash.zh-TW.md) 一樣：
在 512 選 10 的路由之下，一個 512-token 的 ubatch 幾乎會碰到一層裡的每一個 expert，
所以每個 ubatch 都要把整層從 DDR5 讀一遍，而這個成本是攤提在這一批有幾個 token 上的。
拉高 `-ub` 就是那根槓桿。但 compute buffer 是在載入時按 `-ub` 配置的，
而這張卡的顯存早就被訂光了。

15,505 token 的真實文章 prompt，`-c 24576`，`-b` = `-ub`，`--threads 16`：

**在 `-ncmoe 28` —— decode 最好的擺法 —— 512 以上一格都裝不進去。**

| ub | 峰值顯存 | Prefill | TTFT | 生成 | |
|---|---|---|---|---|---|
| 512 | 31,963 | 254 | 61.0 s | 31.79 | PASS |
| 1024 | 32,101 | — | — | — | PREFILL_OOM |
| 2048 | 32,103 | — | — | — | PREFILL_OOM |
| 4096 | — | — | — | — | LOAD_OOM |
| 8192 | — | — | — | — | LOAD_OOM |

一個 15K 的 prompt 要等 61 秒才出第一個 token，而你只有 146 MiB 的餘裕能對它做任何事。

**在 `-ncmoe 36` —— 八層搬回 CPU —— 曲線就出現了。**

| ub | 峰值顯存 | Prefill | TTFT | 生成 | 對 512 |
|---|---|---|---|---|---|
| 512 | 22,471 | 202 | 76.6 s | 27.75 | — |
| 1024 | 22,755 | 323 | 47.9 | 27.68 | 1.6× |
| 2048 | 23,025 | 520 | 29.8 | 27.49 | 2.6× |
| 4096 | 23,567 | 790 | 19.6 | 27.77 | 3.9× |
| **8192** | **25,453** | **1,040** | **14.9** | 27.66 | **5.1×** |

**5.1× —— 跟 DeepSeek 一模一樣的倍數**，代價是 2,982 MiB 的 compute buffer
（DeepSeek 是 3.5 GB）。生成全程平在 27.5–27.8，這正是讓 prefill 那一欄沒有爭議的對照組。

所以單卡上的取捨是：

| | `-ncmoe 28`，ub 512 | `-ncmoe 36`，ub 8192 |
|---|---|---|
| 生成 | **31.8** | 27.7（−13%） |
| Prefill | 254 | **1,040（4.1×）** |
| 15.5K prompt 的 TTFT | 61 s | 15 s |

把八層 expert 搬下卡換到 3 GB，3 GB 換到四倍的 prefill，代價是 13% 的 decode。
要哪一邊完全看 prompt 長度；只要有 system prompt 加一點歷史，就是右邊。

## 兩張卡

Engram 不管怎樣都留在 RAM（上面每一輪它都沒離開過）。
其餘是 87.24 − 26.82 = **60.42 GiB**，而兩張 5090 名目上有 64 GB。所以裝得下嗎？

把 vLLM 實例停掉的期間，`CUDA_VISIBLE_DEVICES=0,1`、`-c 8192`、`--threads 16`：

| 配置 | GPU 0 | GPU 1 | 生成 | ms/token | |
|---|---|---|---|---|---|
| `-ncmoe 0`，預設 split | — | — | — | — | **載入失敗，GPU 0 差 66 MiB** |
| `-ncmoe 2`，預設 split | 30,621 | 30,721 | 82.77 | 12.08 | PASS |
| `-ncmoe 2`，`--tensor-split 25,23` | 31,845 | 29,495 | 83.72 | 11.94 | PASS |
| `-ncmoe 4`，`--tensor-split 26,22` | 30,633 | 28,273 | 73.02 | 13.69 | PASS |
| `-ncmoe 0`，`--tensor-split 25,23` | — | — | — | — | 載入失敗 |

**差一點 —— 但留兩層在 CPU，預設 split 就有 82.8 tok/s，是單卡 35.53 的 2.33 倍**
（手動指定 split 是 83.7、2.36 倍，但那在別處要付代價，見下）。
而且單卡掃出來的定律跨過 PCIe 一點都不用調：
`-ncmoe 4` 預測 13.69 ms，實測 13.69；`-ncmoe 2` 預測 12.5，實測 11.94–12.08。
最後那兩層是 expert bytes 的 4%，每 token 約 1.2 ms ——
從 83 走到 88 tok/s 的地板是 6% 的增益，不是「全部進 GPU」暗示的那種飛躍。

### 為什麼 60 GiB 裝不進 64 GB

名目數字漏掉了三件事，全部實測過：

1. **驅動保留。** 每張卡回報 32,607 MiB，能配置 32,109
   （保留 498 MiB，DeepSeek 那頁量的）。兩張卡：64,218 MiB，62.7 GiB。
2. **每卡各付一份開銷。** CUDA context、compute buffer、KV 是**每張卡**各付一次。
   單卡 `-ncmoe 48` 用了 6,487 MiB，其中權重 5,100 —— `-c 8192` 之下 1,387 MiB 的開銷。
   兩張卡約 2,400。
3. **層的顆粒度。** Split 只能整層整層地移，每層 1,274 MiB，所以總有一張卡扛著那個餘數。

```
可配置        64,218
權重          61,870   (60.42 GiB)
每卡開銷     ~2,400
                  ──────
                ~ −50 MiB
```

這不是總量的問題；是 GPU 0 自己超了 66 MiB ——
而那是失敗的那筆 allocation 的大小，所以真正的缺口是**至少** 66。

**q8_0 KV 補不上。** 12 層 full attention、2 個 KV head，f16 每 token 24 KiB，
q8_0 約 14.8（含它那個每層的暫存區 —— 見 [kv-cache-quant.zh-TW.md](../docs/kv-cache-quant.zh-TW.md)）。
`-c 8192` 之下兩卡合計省 74 MiB，短的那張卡分到 37。沒實測，但算式離得太遠。

**小一號的量化補得上。** 從 repo 裡另外兩個 build 的 GGUF header 讀出來的，不用下載：

| | 檔案 | expert bpw | 非 engram | `-ncmoe 0` 每卡 | 餘裕 |
|---|---|---|---|---|---|
| `UD-IQ4_XS`（這次） | 87.2 GiB | 3.942 | 60.42 GiB | 32,135 | **超** |
| **`UD-Q3_K_XL`** | 83.8 | 3.697 | **56.97** | 30,369 | **1.7 GB** |
| `UD-IQ3_XXS` | 76.3 | 3.220 | 49.50 | 26,544 | 5.5 GB |

三個檔案的 engram 都是 `IQ4_NL`、26.82 GiB —— unsloth 把它釘死了 ——
所以省下來的全落在 expert 上，而那正是要塞進去的部分。
`UD-Q3_K_XL` 讓 `-ncmoe 0` 進得了兩張卡、還剩 1.7 GB；
`UD-IQ3_XXS` 剩到夠放像樣的 context 或大的 `-ub`，但 expert 是 3.22 bpw，
[model-quant.zh-TW.md](../docs/model-quant.zh-TW.md) 把它歸在「只在什麼都裝不下時用」。
沒有 `IQ4_XXS` 這種東西；ggml 的 IQ4 家族只有 `_XS` 和 `_NL`。

### 預設的 split，以及 `-ncmoe` 從哪張卡卸貨

進去之前的預測 —— 根據 [gotcha #8](../docs/gotchas.zh-TW.md) —— 是
`-ncmoe 2` 配預設 split 會讓 GPU 1 拿到 24 層 expert、GPU 0 拿 22 層，重的那張會 OOM。
它沒有：**`-ncmoe 2` 之下預設 split 配平到 100 MiB 以內**（30,621 / 30,721），
手動指定的 `25,23` 反而讓它們差了 2,350。

但那個平衡不是普遍性質。在底下的 prefill 掃描裡，GPU 1 在 `-ncmoe 4` 和 `-ncmoe 2`
的數字**完全相同** —— 每個 `-ub` 都是 31,295、31,763、32,015 —— 而 GPU 0 掉了 2,560。
**邊界不跟著 `-ncmoe` 動，`-ncmoe` 拿掉的每一層都是從 GPU 0 拿的。**
所以 `-ncmoe 4` 之下預設 split 讓 GPU 0 空著 2.6 GB、GPU 1 滿的 —— 正是 gotcha #8
描述的形狀 —— 這也是 `-ub 2048` 在兩個設定下都死在 device 1 的原因。
`-ncmoe 2` 那次的平衡是 splitter 算出來的還是那條邊界的巧合，這裡沒有釐清。

實用規則：`-ncmoe N` 要配一個把層**搬到 GPU 0** 的 `--tensor-split`。
`-ncmoe 4 --tensor-split 26,22` 在上面量到 30,633 / 28,273，
那會給 GPU 1 留下 `-ub 2048` 要的那約 1 GB。這個組合沒測。

### 兩張卡：prefill

第二張卡的價值在這裡。同一個 15,505 token 的 prompt，`-c 24576`，預設 split：

| `-ncmoe` | ub | GPU 0 | GPU 1 | Prefill | TTFT | 生成 | |
|---|---|---|---|---|---|---|---|
| 2 | 512 | 31,199 | 31,295 | **1,199** | **12.9 s** | 67.3 | PASS |
| 2 | 1024 | 31,657 | 31,763 | **1,527** | **10.2 s** | 66.1 | PASS |
| 2 | 2048 | 32,085 | 32,015 | — | — | — | PREFILL_OOM，device 1 差 1,006 MiB |
| 2 | 4096 | — | — | — | — | — | LOAD_OOM |
| 4 | 512 | 28,639 | 31,295 | 927 | 16.7 s | 59.6 | PASS |
| 4 | 1024 | 28,975 | 31,763 | 1,249 | 12.4 s | 59.9 | PASS |
| 4 | 2048 | 29,317 | 32,015 | — | — | — | PREFILL_OOM，device 1 差 1,006 MiB |
| 4 | 4096 | — | — | — | — | — | LOAD_OOM |

**預設的 `-ub` 512 之下，兩張卡的 prefill 是 1,199 tok/s，單卡是 254 —— 4.7 倍，
15K prompt 的第一個 token 從 61 秒變 13 秒。** 這還沒動到 `-ub` 那根槓桿：
expert 進了顯存之後，DDR5 攤提的問題大部分消失，batch 大小就沒那麼要緊了。
`-ub` 1024 再加 27%；2048 裝不下，原因就是上面 GPU 1 那件事。

對照單卡最好的 prefill 配置：

| | 1× 5090，`-ncmoe 36`，ub 8192 | 2× 5090，`-ncmoe 2`，ub 1024 |
|---|---|---|
| Prefill | 1,040 | **1,527** |
| TTFT，15.5K | 14.9 s | **10.2 s** |
| 生成（填了 15.5K） | 27.7 | **66.1** |

兩張卡的機器兩個軸同時贏，單卡做不到 —— 在單卡上 prefill 和 decode 搶的是同一塊 3 GB。

這張表還說了兩件事。每一層 CPU expert 在 `-ub` 512 之下讓 prefill 每 token 多約 0.12 ms
（`-ncmoe 2` → `4` 是 1,199 → 927），跟單卡掃描給的每層數字一樣（八層 254 → 202）——
這是 DDR5 讀取的性質，跟卡的數量無關。而這裡的生成是 67，不是 `-c 8192` 短 prompt
量到的 83：視窗裡 15.5K token 的 KV 吃掉約 20%，比單卡掉的約 10% 多，
因為隨 context 長大的 GPU 側地板在 12 ms 的 token 裡佔的比例，比在 28 ms 的裡大。

把每層成本外推到 `-ncmoe 0`，`-ub` 512 約 1,700 tok/s ——
`UD-Q3_K_XL` 全部進兩張卡時預期能到的數字。外推，不是實測。

### 解開 `-ub` 2048 的那個 split

`-ncmoe 4 --tensor-split 26,22`，掃描指向的那個組合：

| ub | GPU 0 | GPU 1 | Prefill | TTFT | 生成 | |
|---|---|---|---|---|---|---|
| **2048** | 31,963 | 29,981 | **1,548** | **10.0 s** | 59.3 | PASS |
| 4096 | 32,099 | 29,975 | — | — | — | PREFILL_OOM，device 0 |

如預測般成立 —— GPU 1 有了空間、2048 載得進去 —— 而且是這一頁最高的 prefill 數字，
高 1.4%，代價是比 `-ncmoe 2` / `-ub` 1024 少 7 tok/s 的 decode。不值得；1,527 / 66 還是那個配置。
`-ub` 4096 死在 GPU 0 —— split 之後它成了滿的那張 —— 死在下一節要講的那個配置上。

## Context 天花板

### 兩張卡，階梯看到的樣子

ctxprobe 在 `-ncmoe 4 --tensor-split 26,22`、f16 KV、`--reasoning-budget 0`、
`--max 131072`。它的 `PEAK_VRAM` 欄只讀 `CUDA_VISIBLE_DEVICES` 的第一個裝置，所以這裡是 GPU 0。

| Context | 結果 | GPU 0 | Prefill | 生成 | 填充 |
|---|---|---|---|---|---|
| 8,192 | PASS | 30,687 | 1,732 | 59.2 | 4,024 |
| 131,072 | LOAD_FAIL | — | — | — | — |
| 69,632 | PASS | 31,721 | 1,785 | 59.2 | 4,024 |
| 100,352 | LOAD_FAIL | — | — | — | — |
| 84,992 | PASS | 31,985 | 1,821 | 59.6 | 4,024 |
| 92,672 | PREFILL_OOM | 32,103 | — | — | died@4024 |
| 88,832 | PASS | 32,053 | 1,823 | 58.0 | 4,024 |
| 90,624 | PASS | 32,083 | 1,810 | 58.7 | 4,024 |
| 91,648 | PASS | 32,099 | 1,812 | 59.2 | 4,024 |
| 92,160 | PASS | 32,107 | 1,819 | 58.8 | 4,024 |
| 92,416 | PREFILL_OOM | 32,105 | — | — | died@4024 |

然後是真正重要的部分：

```
Validating 92160 at 95% fill...
  92160 failed at 95% fill (PREFILL_OOM)      died@87063
  91904 failed at 95% fill (PREFILL_OOM)      died@86816
  91648 failed at 95% fill (PREFILL_OOM)      died@86592
  still failing at 95% fill after backing off.
  The ladder stops at 4096 tokens, so this is a size only a near-full
  prompt breaks — no such case has been measured before.

Largest context that cleared the ladder: 91648 tokens (UNCONFIRMED)
```

**上面那張表裡約 63K 以上的每一個 PASS 都是假的**，而 95% 驗證是唯一說出這件事的東西。
這個分支從階梯寫好那天就一直在工具裡，從來沒被走到過。這是第一次。

### 壞在哪，以及為什麼階梯看不到

這幾次失敗共用一個 backtrace，而它不是這個專案先前每一次失敗的那一個：

```
ggml_cuda_op_top_k
  -> argsort_f32_i32_cuda_cub
    -> ggml_cuda_pool_vmm::alloc          CUDA error: out of memory
```

那是 **QSA indexer 的 top-k 選擇**。它替 ubatch 裡的每一個 query 列對每一個已快取的 block
（每 block `compress_ratio` 4 個 token）打分數、排序、留下 `indexer.top_k` 2048 個。
分數 tensor、它的 graph buffer、排序的暫存區，全部隨 **n_kv × n_ubatch** 長大 ——
隨著快取裡已經有多少 context。在 `top_k + compress_ratio − 1` = 2,051 個 token 以下
選的是全部，QSA 在結構上就是 dense；超過之後，工作集隨視窗裡的每一個 token 長大。

階梯的三階是 64、512、4096。4,096 token 的 prompt **有**走到 sparse 路徑 ——
但把它配置成 n_kv ≈ 4K 的大小。87K 的 prompt 要 20 倍，在一張只剩 10 MiB 的卡上，
它在進行到約 5,000 token 時死掉，離 4096 那階通過只有 2.9 秒。

這是這個專案第一次量到**隨 prompt 長度長大**的 buffer。
[gotcha #4](../docs/gotchas.zh-TW.md) 說它們不會 —— 階梯之所以成立，
是因為 buffer 跟的是 batch 形狀不是 context —— 對 dense attention 它是對的。
帶學習型 indexer 的稀疏注意力是例外。這對工具有兩個後果：在這個架構上 4096 那階
不能當作滿視窗的代理，所以搜尋該把 fill-validated 的天花板二分出來，而不是退兩步就停 ——
ctxprobe 現在會這樣做了，下一小節就是它的第一次執行。

但這**不**代表這個專案上每一個稀疏注意力的天花板都是錯的。
[DeepSeek-V4-Flash 的 131,072](rtx5090-32gb-deepseek-v4-flash.zh-TW.md)
階梯驗證過、從未填充驗證，所以它是最明顯的下一個嫌疑犯；
填滿之後它通過了（124,075 個 token，數字沒變）。
它的 indexer 很可能有同樣的成長性質，但那個配置的峰值是 15,469 / 32,109 ——
有 16 GB 的餘裕讓它去長，而這裡只有 10 MiB。
**會隨視窗長大的 buffer，只有在卡滿的時候才會變成天花板。**

### 把真正的天花板夾出來

ctxprobe 退兩步就停了。要找到 fill-validated 的天花板真正在哪，
同一個擺法直接餵 95% 的 prompt，長度用伺服器自己的 `/tokenize` 量出來：

| `-c` | ub | prompt token | GPU 0 | GPU 1 | Prefill | TTFT | |
|---|---|---|---|---|---|---|---|
| **61,440** | 512 | **57,819** | **32,055** | 30,171 | **621** | **93 s** | **PASS** |
| 69,632 | 512 | 66,041 | 32,107 | 30,215 | — | — | graph 重新保留，device 0 要 1,203 MiB |
| 77,824 | 512 | 72,439 | 32,105 | 29,965 | — | — | top-k argsort |
| 91,648 | 256 | 86,791 | 32,107 | 29,869 | — | — | top-k argsort |
| 91,648 | 128 | 86,791 | 32,107 | 29,891 | — | — | top-k argsort |

**61,440 在 95% 填充下撐住了，GPU 0 剩 54 MiB。69,632 沒有。**
這個擺法的確認天花板在兩者之間，從餘裕和每 token 的成長看，大概在下界往上 2K 以內。
這一頁要記的數字是：**兩張卡、f16 KV，約 61K** —— 對照階梯宣稱的 91,648。

69,632 死得跟其他幾個不一樣 —— 不在 pool 裡，而在 graph allocator：
prefill 進行到一半、n_kv 長大時，它在 GPU 0 重新保留一塊 1,203 MiB 的 compute buffer。
同一個原因，另一個配置器。

填了 58K 之後的 prefill 是 621 tok/s，同一個擺法在 15.5K 是 927：
視窗越滿，indexer 在每個 ubatch 裡佔的比例越大。

**`-ub` 那個假設這輪沒有測到。** 91,648 那兩列本來是要驗證更小的 ubatch
能不能把 indexer 的工作集縮到夠買回 context —— 兩個都死得跟 512 一模一樣。
但在 91,648，光 GPU 0 上的 KV 就比 61,440 多約 370 MiB，餘裕只有 54；
那個擺法在 sparse buffer 進場之前就先被 KV 卡死了，`-ub` 再小也顯示不出任何東西。
該做的測試是 69,632 配 `-ub` 128。還開著。

### 單卡，用修好的工具

ctxprobe 現在會做上一小節用手做的事：勝出者在 95% 填充下失敗時，
在它底下二分，每個探測點都用完整長度驗證，把撐得住的大小跟階梯宣稱的並排回報。
這條程式路徑的第一次執行，在唯一空著的那張卡上 ——
`-ncmoe 30`、`-ub 2048`、f16、`--min 32768 --max 81920`：

```
32768      PASS        30253       1101.01    29.62      4024
81920      PASS        32029       1113.98    30.18      4024

Validating 81920 at 95% fill...
  81920 failed at 95% fill (PREFILL_OOM)
  Bisecting for the size that holds a full window:
  57088 failed   44800 holds   50944 failed   47872 holds
  49408 failed   48640 failed  48128 holds    48384 failed
  confirmed: 48128 — the ladder had said 81920, 33792 tokens too high

Largest context that actually runs: 48128 tokens
  peak VRAM 32097 MiB | prefill 654.45 tok/s | generate 26.10 tok/s
```

**階梯把天花板高估了 41%。** 八個填充驗證的探測點，每個都是一次完整啟動加 40–55K token 的 prefill，
約半小時收斂到 256 的步進。那就是在這個架構上誠實數字的代價，而且它是有界的 ——
範圍的 log₂ —— 舊的退兩步做法會停在 81,408 UNCONFIRMED。

所以單卡的 fill-validated 天花板是 **48,128，在 `-ncmoe 30` / `-ub 2048` / f16 之下**
（填了 45,571 個 token，prefill 654 tok/s，decode 26.1）。
`-ub 2048` 讓 indexer 的工作集比 512 大四倍，所以預設 `-ub` 之下這個擺法撐得更多；48,128 是它的下界。

## Engram 放 SSD

問題是：expert 在 `-ncmoe 28`（CPU 端 32.3 GiB）之下，
那 26.8 GiB 的 engram 能不能留在 NVMe 而不是 RAM？

### 沒有針對單一 tensor 的開關

llama.cpp 把整個檔案 mmap 進來。mapping 存在的時候，
沒有任何方法能只踢掉一個 tensor 的頁面 —— 實測，512 MiB `MAP_PRIVATE`、全部 touch 過：

| 動作 | 之後還常駐 |
|---|---|
| 從另一個 fd `posix_fadvise(DONTNEED)` | **100%** |
| 對 mapping 本身 `madvise(MADV_DONTNEED)` | **100%** |
| 先 unmap，再 `posix_fadvise(DONTNEED)` | 0% |
| 在 cgroup `MemoryMax=192M` 之下 touch | 37% |

核心的 `invalidate_mapping_pages()` 會跳過被 map 住的頁。
只有記憶體壓力下的 reclaim 會把它們趕走，所以這個實驗是**整個 process 的 cgroup 上限**，
賭的是 LRU 會保留每個 token 都碰的 expert 頁、丟掉 engram。

`--no-mmap` 會靜靜毀掉這件事：權重變成匿名記憶體，上限會把它們送進 swap（這台 7 GB），
而不是送回 GGUF。

### 上限是鈍器，而且看得出來

變化的 prompt（每個 220 個字、從 50 個字的清單抽，所以 trigram 是新的），
13 個請求，每個約 285 個 prompt token + 64 個生成 token。
每一組之前都先清 cache，讓頁面記在對的 cgroup 上。
時間取第 13 個請求；磁碟是 `/proc/pid/io` 對全部 4,472 個 token 的總和。

| 上限 | File pages | expert 常駐？ | 生成 | Prefill | 磁碟 / token |
|---|---|---|---|---|---|
| 無 | 87.3 GiB | ✅ | 34.56 | **282** | 0 |
| 46G | 39.2 | ✅（需要 32.3） | 34.05 | **217** | 3.13 MiB |
| 42G | 35.2 | ✅ | 34.31 | 217 | 4.81 MiB |
| 38G | 31.2 | **❌ 抖動** | 28.64 | 157 | 4.47 MiB |

最後一列要讀成一組失敗的實驗，不是一個數據點：
file pages 低於 expert 需要的 32.3 GiB，付的是 expert 的 miss，engram 沒辦法從裡面拆出來。

兩組乾淨的實驗一致指向三件事：

**Decode 幾乎沒感覺。** 34.05 和 34.31 對 34.56。生成時 engram 每 token 的 16 列
要嘛已經在快取裡、要嘛是 16 次約 54 µs 的 fault，而 16 × 54 µs 是 28 ms 裡的 0.9 ms。
底下的穩態實驗把真實數字定在 −3.6%。

**Prefill 要付錢。** 第 13 個請求是 217 對 282。但第 13 個請求不是穩態 ——
見下面的軌跡，以及再之後的真實文章實驗。

**磁碟讀取是權重的 50 倍。** 每 token 3.13 MiB，也就是每取一列讀 200 KiB，
而一個 4 KiB 的 page 裡只有 90 bytes 的 `IQ4_NL` 是要的。這顆碟的 `read_ahead_kb` 是 128，
`llama-mmap.cpp` 對整個檔案宣告了 `POSIX_FADV_SEQUENTIAL`，這會把 readahead 視窗加倍；
fault-around 再加 64 KiB。每次 fault 200 KiB 就是這個算式的答案。
對「按順序讀一次」的 dense 權重來說這是對的；對一張 3.2 億列隨機存取的表，
它每讀一個有用的 page 就拖進 50 個沒用的 —— 這也是為什麼 42G 那組讀得比 46G **更多**：
被吹大的足跡在更少的 slack 裡翻得更快。

### 那是一條 warm-up 曲線，不是穩態

46G 那組逐請求：

| 請求 | Prefill | 生成 |
|---|---|---|
| 0 | 10.8 | 20.6 |
| 1 | 78.9 | 25.9 |
| 2 | 152.3 | 32.0 |
| 3 | 162.9 | 33.5 |
| 6 | 187.5 | 32.7 |
| 9 | 204.1 | 34.6 |
| **12** | **217.4** | **34.1** |

生成到第三個請求就穩了 —— 那是清 cache 之後 expert 分頁回來的過程。
Prefill 到第十三個還在爬。一部分是 50 個字的詞表：只有 2,500 種 bigram，
所以 bigram head 的工作集會在這一輪裡逐漸收斂，真實文章不會這樣。
那個 −23% 是一條曲線上的一個點，而這一輪沒有跑到曲線的盡頭。真實文章的長跑在下面。

### 整個檔案開 `MADV_RANDOM` 會怎樣（別做）

`llama-mmap.cpp` 只在 `if (numa)` 之下套 `POSIX_MADV_RANDOM`，
而 `ggml_is_numa()` 是 `n_nodes > 1` —— 在這台單 socket 的機器上不管有沒有
`--numa distribute` 都是 false，重跑的結果逐位元組相同。
所以用一行補丁強制那個分支，以環境變數控制、否則完全不動：

```diff
-        if (numa) {
+        if (numa || getenv("LLAMA_MMAP_RANDOM")) {
```

同樣的 46G，開著它：

| 請求 | Prefill | 生成 |
|---|---|---|
| 0 | 3.4 | 8.3 |
| 3 | 17.4 | 4.5 |
| 6 | 32.8 | 18.8 |
| 9 | 128.0 | 31.9 |
| 12 | 185.9 | 34.1 |

磁碟讀了 80 GB（原本 14 GB），warm-up 慢五倍。把 engram 的 readahead 關掉，
也同時關掉了 expert 的 —— 32 GiB 要一次一個 4 KiB 的同步 fault 分頁回來 ——
而 expert 對 readahead 的需要，遠大於 engram 被它傷害的程度。
真要修，得是針對單一 tensor 的：只對 engram 的 byte range 做 `madvise(MADV_RANDOM)`，
而且既然 hash 是在 gather **之前**就在 host 端算好的，
可以對 16 × N 列一次全部 `MADV_WILLNEED`，讓核心把 fault 重疊處理而不是一次一個。
兩者這裡都沒測。

### 真實文章上的穩態

合成詞表會收斂，真實文章不會。每組 30 個請求，取 llama.cpp 自己的 `docs/*.md`，
切成約 1,300 字元一段（prompt 316–771 token，中位數約 365），每個生成 48 token，
兩組用同樣的段落、同樣的順序。數字是第 10–29 個請求的平均；
前十個在有上限那組是 warm-up，在沒上限那組是平的。

| | RAM（無上限） | SSD（46G 上限） | Δ |
|---|---|---|---|
| Prefill | 203.7 tok/s | **134.3 tok/s** | **−34%** |
| 生成 | 35.1 tok/s | **33.9 tok/s** | **−3.6%** |
| 每請求牆鐘時間 | 3.15 s | 4.2 s | +33% |
| 每請求磁碟 | 0 | 約 850 MiB | — |
| File pages | 87.3 GiB | 37.0 GiB（需要 32.3） | expert 常駐 |

有上限那組第 10–29 個請求的 prefill 在 91–205 之間、沒有上升趨勢，
所以這是穩態，不是又一個 warm-up 曲線上的點。
沒上限那組同一段請求是 175–273 —— 兩組的 prefill 速率都跟著 prompt 長度走 ——
所以比較用的是同樣 prompt 上的平均值比，不是兩個單一數字。

**Decode：−3.6%。** 每 token 16 列，多數在快取裡，其餘每次一個 54 µs 的 fault。
這是從這顆碟的隨機讀延遲推出來的預測，而它成立了。

**Prefill：−34%。** 一個 365 token 的 prompt 在第一層的 attention 能跑之前要先取 5,840 列。
兩筆成本疊在一起：5,840 次約 54 µs 的同步 fault 約 0.3 s，
5,840 × 200 KiB 的 readahead 約 1.1 GB、以幾 GB/s 算又是 0.3–0.4 s。
對照 1.8 s 的 prefill，那就是實測到的懲罰 —— 而且它說兩個機制貢獻差不多，
這對修法很重要，因為 `MADV_RANDOM` 只拿得掉第二個。

**磁碟：每 token 約 2 MiB，換 1.4 KB 的權重。** 還是那個 readahead 倍數。
真實文章的工作集比 50 個字寬得多，所以 7 GiB slack 裡的翻動永遠不會停。

所以「這個模型的 29% 能不能住在 NVMe 上」的答案是：
**decode 可以，付 4%；prefill 在現在的 mmap 路徑下只剩三分之一的速度。**
省下的是 26.8 GiB 的 RAM。在 186 GB 的機器上這什麼都換不到；
在 64 GB 的機器上，它是這個模型載得進來跟載不進來的差別。

## 方法：這個量測在成功之前騙了我五次

每一次都產出一張乾淨的表，裡面沒有任何訊號。

1. **`curl -s …/health && ready` 在 HTTP 503 也會過。** curl 對任何回應都回 0。
   port 一綁上檢查就通過了，模型還在載入；每個請求都拿到 503；
   六個配置「通過」，時間欄全空。就緒的定義是一個真的生成回 200，沒有別的。
2. **`setsid systemd-run --scope` 掛不上去。** 程序落在 SSH session 的 scope 裡，
   `memory.max = max`。三個上限，RSS 一模一樣。transient 的 `--unit` service 掛得上；
   而且在量任何東西之前，先從 `/sys/fs/cgroup/…/memory.max` 把上限讀回來確認。
3. **Page cache 記在第一個 fault 它的人頭上。** 那 87 GiB 早在前幾輪就進了快取、
   屬於一個舊的 cgroup；新的那個只有 1.4 GiB，上限沒東西可咬。
   每一組之前先清 cache —— 沒人 map 著檔案時 `fadvise` 是有效的 —— 新的 process 才會變成擁有者。
4. **每次同一個 prompt，量到的是快取不是磁碟。** 每個上限都是零磁碟讀取，
   看起來像「engram 放 SSD 免費」。其實是「同樣的 trigram 第二次免費」。換成變化的 prompt 才修好。
5. **一個把 expert 也趕走的上限，量到的是 expert。** 34G 留下 27 GiB 的 file pages，
   expert 要 32.3 GiB，回報 −38%。file pages 對 expert bytes 是那道檢查；上面每張表都有它。
6. **50 個字的清單跑 13 個請求，是一條 warm-up 曲線。** 2,500 種 bigram 逐漸收斂進快取，
   prefill 一直漲到實驗結束 —— 那個 −23% 只是實驗剛好停在哪裡。
   真實文章、30 個請求、後 20 個平均：−34%，而且是平的。
7. **「四個字元一個 token」不是 tokenizer。** 用字元數估 95% 的 prompt，
   在循環過的文字上超了 6–9%；五次跑都在任何 prefill 開始之前就拿到 HTTP 400
   `exceeds the available context size`，而腳本把五次全部歸成 PREFILL_OOM。
   ctxprobe 用伺服器的 `/tokenize` 逼近就是為了這個；繞過它的捷徑本身就是 bug。
8. **`curl` 回來之後再 `pgrep` 是個競態，而 `-o` 在連線斷掉時不會清空檔案。**
   在 prefill 中途死掉的伺服器先關 socket、後退出；卡在那個縫隙裡檢查它是「活的」，
   而輸出檔裡還是上一輪的 JSON。四次 crash 被歸成「REJECTED」，還附著一個過期但看起來
   合理的 body。伺服器 log 裡的 backtrace 是唯一搶不贏的證人。

## 尚未量測

- 用會二分的 ctxprobe 把兩張卡的天花板收斂到 256 步進，在 61,440 和 69,632 之間；
  以及 q8_0 KV 之下的同一個數字。
- 69,632 配 `-ub` 128：在稀疏注意力模型上，更小的 ubatch 能不能拿 prefill 換 context。
