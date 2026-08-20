# RTX 5060 Ti 16GB — Qwen3.8-27B

[English](rtx5060ti-16gb-qwen3.8-27b.md) · **繁體中文**

實測於 2026-08-20，用的是產出
[Qwen3.6-27B 那組數字](rtx5060ti-16gb-qwen3.6-27b.zh-TW.md)的同一張卡。
同架構、同工具、同方法 —— 所以這是一組對照實驗，不是兩個各跑各的 benchmark。

第一眼看起來像 context 翻倍。它不是。

## 硬體

刻意跟 Qwen3.6 那輪完全一致。

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 回報 16311 MiB，可配置 15849 MiB** |
| 驅動 | 595.84 |
| llama.cpp | `0a50d99`（2026-07-23），CUDA build |
| 主機 | i7-14700，64 GB DDR4 —— 底下沒有任何數字跟它有關 |

全部跑在顯存裡（`-ngl 999`，沒有 CPU offload）。

## 這個模型跟 Qwen3.6 架構完全相同

兩邊都是 `model_type: qwen3_5`，產出的 GGUF `general.architecture` 都是
`qwen35`。逐欄對過：

```
num_hidden_layers        64      full_attention_interval  4
num_attention_heads      24      num_key_value_heads      4
head_dim                 256     hidden_size              5120
vocab_size               248320  max_position_embeddings  262144
```

llama.cpp 不用升級 —— 現有的 build 本來就有 `qwen35`。

只有一個差別，而它在後面很重要：**Qwen3.8 的 GGUF 多帶了第 65 個 block。**

```
qwen35.block_count            = 65     （Qwen3.6：64）
qwen35.nextn_predict_layers   = 1      （Qwen3.6：沒有這個欄位）
```

`blk.64` 是一個完整的 transformer block —— `attn_q`/`attn_k`/`attn_v`、
`ffn_up`/`gate`/`down`，加上 `nextn.eh_proj`、`nextn.enorm`、`nextn.hnorm` ——
合計 **0.425 B 參數**。它是 multi-token prediction 的頭。

而且它是無條件被載入的。`src/models/qwen35.cpp` 只有在遇到**專用的 MTP 檔案**
時才會把主幹設成可選：

```c
const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
```

這個 GGUF 有 `blk.0.attn_norm.weight`，所以 `mtp_only` 是 false，主幹和 `blk.64`
都以必要張量的身分載進去。但 MTP 的**計算圖**是另一個獨立進入點，
一般的 `llama-server` 解碼永遠不會呼叫它。

**代價：約 350 MiB 的權重，換一個不會被執行的 block。** 它**不**吃 KV cache ——
底下的斜率量測就是證據，我們自己先前的一個猜測也是死在那裡。

## 量化的名字決定不了 bits

Qwen3.8 的 GGUF repo 沒有純 `IQ4_XS`，只有 `UD-IQ4_XS`。
把它當成 Qwen3.6 那個 `IQ4_XS` 的接班人是最直覺的做法，而且是錯的。

bits-per-weight 是用 ggml 各型別實際的 block 大小算出來的，不是估的：

| | 參數量 | tensor bytes | **bpw** |
|---|---|---|---|
| Qwen3.6 `IQ4_XS` | 26,895,998,464 | 14,714 MiB | **4.5892** |
| Qwen3.8 `UD-IQ4_XS` | 27,320,697,856 | 13,582 MiB | **4.1703** |
| Qwen3.8 `UD-Q4_K_S` | 27,320,697,856 | 14,636 MiB | **4.4939** |

Qwen3.8 參數**更多**（多了 MTP 那層），檔案卻**更小**，
因為 `UD-IQ4_XS` 比它看似取代的那個東西少了 0.42 bpw。
兩個檔案的打包原則完全不同：

| | Qwen3.6 `IQ4_XS` | Qwen3.8 `UD-IQ4_XS` |
|---|---|---|
| 量化型別數 | 4 | **12** |
| 主體 | IQ4_XS 20.2 B | IQ4_XS 13.6 B + IQ3_S 3.8 B + Q3_K 1.9 B |
| 低於 3 bits | 沒有 | 0.80 B 參數 |
| `token_embd` | Q4_K | **Q3_K** |
| `output` | Q6_K | **Q5_K** |

所以真正有意義的對比對象是 4.4939 bpw 的 `UD-Q4_K_S` ——
跟 Qwen3.6 的 4.5892 差 2.07%。

## Context 天花板

四次完整的二分搜尋，每個勝出者都用填滿窗口 95% 的 prompt 重新驗證過。
`--parallel 1`、`-ngl 999`、`--reasoning-budget 0`。

| 模型 | bpw | KV | **天花板** | 峰值顯存 | Prefill | 生成 | 填充 |
|---|---|---|---|---|---|---|---|
| `UD-IQ4_XS` | 4.1703 | `q8_0` | **61,952** | 15847 | 739.93 | 23.39 | 58,735 |
| `UD-IQ4_XS` | 4.1703 | `f16` | **36,352** | 15835 | 825.34 | 27.04 | 34,331 |
| `UD-Q4_K_S` | 4.4939 | `q8_0` | **34,304** | 15847 | 820.15 | 24.98 | 32,333 |
| `UD-Q4_K_S` | 4.4939 | `f16` | **19,712** | 15833 | 875.09 | 27.23 | 18,681 |

Prefill 和生成是在各自標示的填充深度量的，所以**不能跨列比較** ——
窗口越深，prefill 越慢、往裡面解碼也越慢。

## 等精度之下，Qwen3.8 拿到的 context 略少

| KV | Qwen3.6 `IQ4_XS`（4.5892） | Qwen3.8 `UD-Q4_K_S`（4.4939） | Δ |
|---|---|---|---|
| `q8_0` | 34,816 | **34,304** | **−512** |
| `f16` | 20,224 | **19,712** | **−512** |

兩種模式都剛好落在 Qwen3.6 底下 512 個 token —— 也就是兩個搜尋步進 ——
而且 Qwen3.8 還是兩者之中精度**略低**的那一個。

**表面上那個 61,952 對 34,816（+78%）完全是量化打包造成的假象。**
Qwen3.8 這個模型本身沒有為這張卡帶來任何額外的 context。
把 bits 拉到同一水平，它反而小輸。

剩下的差距來自固定配置，不是每 token 成本。從 f16 的天花板把 KV 扣掉，
Qwen3.6 的固定佔用是 14,559 MiB、Qwen3.8 是 14,582 MiB —— 差 23 MiB，
但 Qwen3.8 的 tensor 在磁碟上還**小** 78 MiB。
也就是約 100 MiB 的非權重配置，這版 llama.cpp 不會逐項印出來，
所以這裡只呈現，不硬套解釋。

## KV cache：量化它到底買到什麼

峰值顯存對 context 的關係線性得驚人 ——
下面是相鄰 PASS 列之間的差，不是擬合出來的線：

| | 實測斜率 | 16 層理論值 | 超出 |
|---|---|---|---|
| `f16` | **63.48 MiB/1K** | 64.0 | 約 0 |
| `q8_0` | **38.09 MiB/1K** | 34.0 | **+4.1** |

f16 的斜率在 `UD-IQ4_XS`、`UD-Q4_K_S`、**以及 Qwen3.6**（用它自己文件裡的表
重新回歸：63.48）三者之間完全一致。
這就是排除「MTP 層有 KV cache」的關鍵 —— 多一層 attention 會讓 f16 變成 68.0。

理論值是 `full attention 層數 × kv_heads × head_dim × 2 (K+V) × 每元素 bytes`：
16 × 4 × 256 × 2 = 每 token 32,768 個元素，f16 每元素 2 bytes → 64.0 MiB/1K，
q8_0 每元素 1.0625 bytes → 34.0 MiB/1K。

**f16 對得上理論，q8_0 超出 4.1 MiB/1K。**
那筆超出是每 token 4,198 bytes，而一層份的 per-token KV 存成 f16 是
4 × 256 × 2 × 2 = 4,096 bytes —— 吻合到 2.5%。
它的形狀就是一個「一次容納一層」的 dequant 暫存區，隨 context 線性長大。
f16 不需要這種東西。

實務上的後果：

| | 每 1K token | 相對 f16 的天花板增益 |
|---|---|---|
| `UD-IQ4_XS` | 63.48 → 38.09 | 36,352 → 61,952（**+70.4%**） |
| `UD-Q4_K_S` | 63.5 → 37.8 | 19,712 → 34,304（**+74.0%**） |
| Qwen3.6 `IQ4_XS` | 63.48 → 約 38 | 20,224 → 34,816（+72.1%） |

**q8_0 省下的是每 token KV 的 40%，不是型別大小暗示的 47%。**
差額就是那個暫存區。它當然還是非常值得開，
但一個「把 KV bytes 除以二」的計算機會高估你實際拿得到的 context。

## Prompt 階梯的第三階終於派上用場

四次搜尋裡有三次是用這張卡一貫的方式死的 ——
死在 512 那一階，也就是第一個完整的 `n_ubatch`：

| 配置 | 死在 | 對應 |
|---|---|---|
| `UD-IQ4_XS` q8_0 62,208 | `died@509` | `n_ubatch` |
| `UD-IQ4_XS` f16 36,608 | `died@509` | `n_ubatch` |
| `UD-Q4_K_S` q8_0 34,560 | `died@509` | `n_ubatch` |
| **`UD-Q4_K_S` f16 19,968** | **`died@4024`** | **`n_batch`** |

[Qwen3.6 那一頁](rtx5060ti-16gb-qwen3.6-27b.zh-TW.md)寫過，
到那時為止量到的每一次失敗都落在前兩階，
所以 4096 那一階雖然在原理上是為了清掉 `n_batch`（2048），
卻從來沒有真的抓到過東西。**19,968 就是那個反例。**
它過了 64、過了 512，死在 4,024。

一個只做到 512 的階梯，會把 19,968 判成 PASS。

另外值得記一筆：512 那一階的失敗橫跨兩種 KV 型別，
也就橫跨了兩種 flash attention 狀態 ——
`q8_0` 會強制開啟 flash attention，`f16` 則停在 `auto`。
失敗點沒有移動，這支持了原本的歸因：
卡住的是 ggml 的 memory pool 隨 batch shape 長大，而不是 attention 的實作。

## 實用配置

天花板就只是天花板，要留餘裕：

```bash
llama-server -m Qwen3.8-27B-UD-IQ4_XS.gguf \
  -ngl 999 -c 57344 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --reasoning-budget 0
```

57,344 大約在實測的 61,952 底下 4,600 個 token，
差不多就是把 PASS 和 PREFILL_OOM 分開的那段 `n_ubatch` compute buffer 餘裕。

要抓哪個檔案，看你在最佳化什麼：

| 想要 | 檔案 | 拿到 |
|---|---|---|
| 最多 context | `UD-IQ4_XS`（4.17 bpw） | 61,952 |
| 塞得下的最高精度 | `UD-Q4_K_S`（4.49 bpw） | 34,304 |
| 跟 Qwen3.6 對齊 | `UD-Q4_K_S` | 34,304 對 34,816 |

`UD-Q4_K_M`（16.46 GB）塞不下 —— 它只留約 150 MiB 給 KV。

## 重現

```bash
ctxprobe Qwen3.8-27B-UD-IQ4_XS.gguf  --min 32768 --max 98304 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-Q4_K_S.gguf  --min  8192 --max 45056 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-IQ4_XS.gguf --kv f16 --min 8192 --max 40960 -- --reasoning-budget 0
ctxprobe Qwen3.8-27B-UD-Q4_K_S.gguf --kv f16 --min 4096 --max 28672 -- --reasoning-budget 0
```

模型：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)。

## 這一輪替方法論補上的東西

1. **看 bits，不要看名字。** 兩個掛著同一個 Q4 級距的檔案差了 0.42 bpw。
   任何跨模型的 context 比較，跳過這一步就是在量量化器，不是在量模型。
   見 [model-quant.zh-TW.md](../docs/model-quant.zh-TW.md)。
2. **量化 KV 帶著一筆會隨 context 長大的額外開銷。**
   實測 38.09 MiB/1K，理論 34.0 —— 也就是 q8_0 省下的是每 token 成本的 40%，
   不是它的型別大小暗示的 50%。這修正了
   [kv-cache-quant.zh-TW.md](../docs/kv-cache-quant.zh-TW.md)：
   那頁目前把 q8_0 描述成「一半的記憶體」，
   這對 cache 本身成立，但對一段 context 實際吃掉的顯存不成立。
3. **4096 那一階抓得到東西。** 第一次量到第三階的失敗。
