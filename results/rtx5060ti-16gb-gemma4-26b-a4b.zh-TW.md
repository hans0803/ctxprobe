# RTX 5060 Ti 16GB — Gemma 4 26B-A4B（QAT q4_0）

[English](rtx5060ti-16gb-gemma4-26b-a4b.md) · **繁體中文**

實測於 2026-08-11。與 [Qwen3.6-27B 那輪](rtx5060ti-16gb-qwen3.6-27b.zh-TW.md)
同一張卡、同一套工具 —— 這使它成為第一個獨立檢驗：
那邊找到的失敗門檻，究竟屬於 llama.cpp，還是只屬於那個模型。

整輪搜尋：**2 分 39 秒**，十一次啟動。

## 硬體與模型

| | |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 5060 Ti — 回報 16311 MiB，可配置 15849 MiB** |
| llama.cpp | `0a50d99`，CUDA build |
| 模型 | [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)，14.44 GB |
| 設定 | `-ngl 999`、`--parallel 1`、`--cache-type-k/v q8_0`、flash attention 開啟 |

底下所有結果都由兩個架構特性驅動：

- **Sliding-window attention。** 30 層中**只有 5 層是 full attention**；
  其餘 25 層使用 1024 token 的滑動視窗，KV 成本是固定的，不隨 context 成長。
- **Mixture of experts。** 128 個 expert，總參數 26B，
  每個 token 只啟用約 4B。

為什麼選 QAT 版：它是量化感知訓練，而且 14.44 GB 是裝得下的最大變體。
unsloth 的 `UD-Q4_K_S`（16.49 GB）和 `UD-Q4_K_M`（16.95 GB）
在還沒有任何 KV cache 之前就已經超出這張卡。

## Context 上限：105,984

| Context | 結果 | 峰值顯存 | 實際灌入 |
|---|---|---|---|
| 8,192 | PASS | 14363 MiB | 4,038 |
| 69,632 | PASS | 15293 MiB | 4,038 |
| 100,352 | PASS | 15761 MiB | 4,038 |
| 104,192 | PASS | 15821 MiB | 4,038 |
| **105,984** | **PASS** | 15847 MiB | 4,038 |
| 106,240 | PREFILL_OOM | 15845 MiB | died@508 |
| 106,496 | PREFILL_OOM | 15847 MiB | died@508 |
| 107,008 | PREFILL_OOM | 15791 MiB | died@63 |
| 108,032 | PREFILL_OOM | 15807 MiB | died@63 |
| 115,712 | LOAD_FAIL | — | — |
| 131,072 | LOAD_FAIL | — | — |

勝出者在 95% 填充下的驗證：**確認通過，實際灌入 100,194 個 token**。

**在同一張卡上是 Qwen3.6-27B 的三倍 context**，而模型尺寸相當。
原因是 sliding-window attention：30 層中只有 5 層持有「會隨 context 成長」的 KV，
對比 Qwen3.6 的 64 層中有 16 層。

## 門檻不是模型特有的

這才是這一輪真正的收穫。每一次失敗都精確落在
「另一個完全不同架構」上量到的同樣兩階：

| 階 | 常數 | Qwen3.6-27B（dense + linear attn） | Gemma4-26B-A4B（MoE + sliding window） |
|---|---|---|---|
| 64 | `MMQ_DP4A_MAX_BATCH_SIZE` | `died@64` | `died@63` |
| 512 | `n_ubatch` 預設 | `died@512` | `died@508` |

（是 63 和 508 而不是 64 和 512，因為階梯透過伺服器的 tokenizer
把 token 數收斂到目標值的 3% 以內。）

兩個在架構上毫無共通之處的模型 —— dense vs MoE、linear attention vs sliding
window、64 層 vs 30 層 —— 在完全相同的兩個 batch 尺寸上失敗。
門檻屬於 llama.cpp build 與 GPU，正如閱讀 `mmq.cu` 所預測的。
不需要為每個模型重新推導。

另外注意是哪一階抓到哪個 config：106,240 和 106,496（剛過線）能撐到 508，
而 107,008 和 108,032（超出更多）死在 63。
超出上限越多，死得越早。

## 速度

| | Prompt | Prefill | 生成 |
|---|---|---|---|
| 搜尋階梯 | 4,038 tokens | 約 3,510 tok/s | 約 120 tok/s |
| **完整驗證** | **100,194 tokens** | **1,889.81 tok/s** | **50.82 tok/s** |

要引用的是驗證那一列：階梯的數字描述的是幾乎空著的視窗。

與同一張卡上的 Qwen3.6-27B 對比，各自在自己的上限：

| | Gemma4-26B-A4B | Qwen3.6-27B |
|---|---|---|
| 最大 context | **105,984** | 34,816 |
| Prefill（滿載） | **1,890 tok/s** | 860 tok/s |
| 生成（滿載） | **50.8 tok/s** | 24.9 tok/s |
| 檔案大小 | 14.44 GB | 15.44 GB |

大約兩倍的吞吐量，搭配三倍的 context。
生成速度的差距來自 MoE —— 每個 token 只啟用約 4B 參數而不是 27B ——
context 的差距則來自 sliding-window attention。
兩者都不是關於輸出品質的陳述，那不在這個專案的量測範圍內。

生成速度在負載下的表現也值得看：空視窗 120 tok/s、100K 時 50.8 tok/s，
對比 Qwen3.6 的 26 → 24.9。
這裡的相對跌幅大得多，這與「5 層 full attention 仍必須對著 100K 的 cache 做注意力」
是一致的。

## 溢出到 8GB 卡

限制在真實 8GB 卡的預算下（剩餘 7799 MiB，
為什麼 16GB 的 5060 Ti 可以代表 8GB 版本見
[spill-cost.zh-TW.md](../docs/spill-cost.zh-TW.md)）：

| 方法 | 裝得下？ | 生成 |
|---|---|---|
| **`--cpu-moe`**（expert 放系統記憶體） | **可以 —— 只用 2.4 GB 顯存** | **39.7 tok/s** |
| `--n-cpu-moe 15`（一半的層） | 不行 | — |
| `-ngl 20`（dense 式，對照組） | 不行 | — |

**`-ngl` 對 MoE 模型是錯的工具。** 每一層都含有 expert，
所以按層砍並不會把大宗移走。按張量類型搬移才有效：
權重從 14.44 GB 掉到約 2.4 GB。

騰出這麼多顯存之後，這張 8GB 卡跑得動 **131,072 context** ——
那是當時 ctxprobe 的搜尋上限，不是模型的極限 ——
prefill 612 tok/s、生成 24.6 tok/s。上限後來已提高到 262,144；
在重新量測之前，這個數字是一個下界。
一個 26B 模型，在 8GB 上，128K context。

### 為什麼沒有更慢

expert 放在系統記憶體卻還有 39.7 tok/s，
很自然會問一個 4B active 的模型怎麼可能撐得住 DDR4。
路由設定給出答案：

```
expert_count       = 128
expert_used_count  = 8      <- top-8 路由
expert_ff_length   = 704
```

每個 expert 是 `3 × 2816 × 704 = 5.95M` 個參數（gate、up、down）；
以 q4_0 每 32 個權重 18 bytes 計算就是 3.19 MiB。
每個 token 啟用 30 層 × 8 個 expert = 240 個：

```
240 × 3.19 MiB       = 每個 token 0.80 GB
39.7 tok/s × 0.80 GB = 31.9 GB/s
```

這台主機實測的 DDR4 頻寬：copy 43.1 GB/s（讀+寫）、
單執行緒讀取 27.0 GB/s。
所以 31.9 GB/s 已經接近飽和 —— **它就是頻寬受限**，跟預期一致。

之所以感覺比「4B active 過 DDR4」該有的速度快，
是因為那 4B 裡**只有 1.43B 是 expert**。
其餘約 2.6B —— attention、shared 層、embedding —— 從來沒有離開 GPU。
MoE 在這裡的優勢不只是「啟用的參數少」，
而是「稀疏的那部分剛好就是可以外放的那部分」。

## 復現方式

```bash
ctxprobe gemma-4-26B-A4B-it-qat-q4_0.gguf --min 8192
```

`--max` 來自 GGUF metadata（262144，被當時的 ctxprobe 截到 131072；現在的上限是 262144）。
不需要 `--reasoning-budget` —— Gemma 4 不是 thinking 模型，
過程中也沒有出現任何 `SILENT` 判定。
