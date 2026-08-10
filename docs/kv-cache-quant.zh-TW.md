# KV cache 量化：為什麼你多半該用 q8_0

[English](kv-cache-quant.md) · **繁體中文**

如果你只有一張消費級顯卡、又想要長 context，這是你手上槓桿最大的一個設定。
這份文件說明它做什麼、代價是什麼，用實測數字而不是經驗法則。

## KV cache 是什麼

模型讀你的 prompt 時，每個 token 在每一層 attention 都會產生兩個向量 ——
**key** 和 **value**。這些會被留著，這樣生成第 5000 個 token 時，
就不用把第 1 到 4999 個 token 從頭再讀一次。這個儲存區就是 KV cache。

重點在於：**它會隨 context 長度線性成長，而且跟模型權重一起住在顯存裡。**

```
顯存 = 模型權重（固定） + KV cache（隨 context 成長） + 其他開銷
```

權重是一次性的成本。KV cache 才是決定你拿到 8K 還是 34K context 的那一項。

## 量化它做了什麼

llama.cpp 預設用 16-bit 精度（`f16`）儲存這些向量。
加上 `--cache-type-k q8_0 --cache-type-v q8_0` 就改用 8-bit：
**同樣的 token 數，記憶體減半。**

在 RTX 5060 Ti 16GB + Qwen3.6-27B-IQ4_XS 上的實測：

| KV 型別 | 每 1K token | 這張卡的最大 context |
|---|---|---|
| `f16`（預設） | 約 68 MiB | 見 [results](../results/rtx5060ti-16gb-qwen3.6-27b.zh-TW.md) |
| `q8_0` | 約 34 MiB | **34,816** |
| `q4_0` | 約 17 MiB | 本專案尚未測試 |

省下來的記憶體直接換成 context。模型其他部分完全不變 —— 一樣的權重、一樣的速度。

## 代價是什麼

`q8_0` KV 造成的品質損失小到日常使用中很難察覺，
而且這是目前多數本地長 context 設定的標準做法。
對於「會被加總起來」的數值來說，每個值 8 bits 仍然是相當充裕的精度。

`q4_0` KV 就是另一回事了。劣化開始變得明顯，而且是以一種特定的方式呈現：
模型對 context 深處的東西變得模糊 —— 而那正是你當初開長 context 想要的內容。
真要用之前值得先測過。

兩者都幾乎沒有速度成本。在我們的卡上，不論用哪種 KV 型別，
生成速度都平穩維持在約 26 tok/s；這個 cache 是記憶體問題，不是吞吐量問題。

## 實務建議

**先用 `q8_0`。** 它大致把你的 context 翻倍，換來的品質代價你多半察覺不到。

當你在做對精度敏感的事、而且 context 短到負擔得起時，才考慮 `f16`。
只有在 `q8_0` 仍然裝不下、而且你測過輸出撐得住的情況下，才動用 `q4_0`。

```bash
llama-server -m model.gguf -ngl 999 -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1
```

## 一個不太直覺的地方

KV cache 的成本取決於模型的架構，不只是參數量。
Qwen3.6-27B 有 64 層，但其中只有 16 層使用 full attention ——
另外 48 層是 linear attention，state 大小固定、不隨 context 成長。
所以它的 KV cache 比一般同尺寸的 27B 便宜得多。

這就是為什麼某個模型的數字不能直接套到另一個同尺寸的模型上，
也是為什麼「實測」勝過「估算」。相關的陷阱收在 [gotchas.zh-TW.md](gotchas.zh-TW.md) ——
特別是 `--parallel`，它會悄悄把你的 KV 配置乘以 4，
維持預設值的話這整頁講的東西都會被抵銷掉。
