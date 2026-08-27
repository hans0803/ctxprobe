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
> **尚未量測：** `-ub` 的 prefill 掃描與 context 天花板。
> 底下所有速度數字都是在預設的 `n_ubatch` 512 和 f16 KV 之下。

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

## 尚未量測

- `-ub` 的 prefill 掃描，在 `-ncmoe 28` 之下。
- 用 ctxprobe 量 context 天花板，同樣的配置。
