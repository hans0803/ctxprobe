# 踩坑清單

[English](gotchas.md) · **繁體中文**

會悄悄吃掉你 context 的東西，以及各自的檢查方式。

## 1. `nvidia-smi` 的 total 不是你的預算

一張 16 GB 的卡回報 16311 MiB，但 CUDA 永遠只能配置到 15849 MiB。
中間 **462 MiB 的落差是驅動保留區** —— framebuffer、page table、配置器結構。
任何程序都拿不到。

```bash
python3 -c 'import torch; print(torch.cuda.mem_get_info()[1] // 2**20, "MiB allocatable")'
```

拿 `nvidia-smi` 的 total 去算餘裕，會比現實樂觀約 460 MiB；
以每 1K token 約 34 MiB 換算，等於憑空多出 13K 的 context。

還有另一個症狀：模型載入後即使看起來還「剩」幾百 MiB，
第二個程序根本啟動不了 —— 因為**光是一個 CUDA context 就要約 138 MiB**。

## 2. `llama-server` 預設開 4 個 slot

`-c N` 是「每個 slot」的值。在預設的 `--parallel 4` 之下，
`-c 8192` 會替 32,768 個 token 配置 KV。

```
load_model: initializing, n_slots = 4, n_ctx_slot = 8192
```

檢查啟動 log 裡的 `n_slots`。單人測試的話，`--parallel 1` 等於免費的 context ——
在我們的測試卡上，光這一項就從 4,096 變成 16,384，其他什麼都沒動。

這個行為在 llama.cpp 不同版本間改過 —— 有些版本把 `-c` 當成總量再分給各 slot。
不要相信這一頁勝過你自己的啟動 log：去讀 `n_ctx_slot` 和 `n_slots` 再相乘。

## 3. `n_ctx` 會被對齊到 256

```c
// src/llama-context.cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

`-c 35000` 會變成 35072。256 才是真正的搜尋粒度；
用 1024 當步進每一階會跳過三個可測的尺寸。可以從 log 的 `n_ctx_slot` 確認。

## 4. 短 prompt 無法認證一個 context 大小

這是最大的一個。較長的 prompt 會實例化短 prompt 從來碰不到的 CUDA kernel，
而第一次載入某個 kernel 需要 device memory —— 一張幾乎滿載的卡拿不出來。
於是會發生這種事：

| Context | 載入 | 18-token prompt | 64-token prompt |
|---|---|---|---|
| 34816 | 成功 | 25.99 tok/s | 正常 |
| 35072 | 成功 | 25.98 tok/s | **CUDA OOM** |

載入過程、健康檢查、小量生成 —— 沒有任何一項能區分這兩者。
**64 個 token 可以** —— 這個門檻遠低於「灌滿視窗」，
而這件事很重要，因為它讓檢查變得便宜。
想確認機制的話用 `CUDA_MODULE_LOADING=EAGER`：
它會把所有 kernel 提前載入，把執行期崩潰變成你不可能忽略的啟動失敗。

64 來自 `MMQ_DP4A_MAX_BATCH_SIZE`，但要注意它外面那層條件：

```c
return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
```

在沒有 FP16 MMA 的卡上，左邊會短路，全部走 dp4a ——
根本不存在 64 這個切換點。這個門檻既是 llama.cpp 的性質，
也同樣是你那張卡世代的性質。

這就是 `ctxprobe` 回報的 `PREFILL_OOM`，而 `died@64` 會指出是哪一階殺死它的。
這個判定以前叫 `LONG_OOM` —— 那是「相信 buffer 會隨 prompt 長度成長」時期的遺留。
它們並不會成長，而那個名字會讓人去找一個根本不存在的記憶體洩漏。

相關的一點：子程序這樣死掉之後會變成殭屍程序（defunct），
而只追蹤自己狀態的上層 gateway 會繼續回報這個部署是健康的。
要檢查程序，不是檢查狀態端點。

## 5. PEAK_VRAM 沒辦法告訴你什麼會失敗

大家第一眼會看的那一欄，正好是回答不了這個問題的那一欄：

| Context | 峰值顯存 | 判定 |
|---|---|---|
| 34,816 | 15845 MiB | PASS |
| 35,072 | 15847 MiB | **CUDA OOM** |

差 2 MiB，結果相反。這不是取樣不夠密的問題，是結構性的：
失敗的那次配置**正因為失敗了**所以從來不會出現在用量裡，
而它要求的量本來就遠低於 `nvidia-smi` 的 1 MiB 解析度。

把峰值顯存讀成「還剩多少餘裕」，永遠不要讀成「離失敗有多近」。
只有判定那一欄能回答後者。

## 6. Thinking 模型會回傳空的 content

Qwen3.6 這類推理模型會把所有東西放進 `reasoning_content`。
在 1200 token 的預算下，模型還在思考就撞到上限，於是 `content` 回傳空字串、
`finish_reason` 是 `length`：

```json
{"message": {"content": "", "reasoning_content": "Here's a thinking process:..."},
 "finish_reason": "length"}
```

要嘛給大得多的預算，要嘛直接關掉 thinking：

```bash
llama-server ... --reasoning-budget 0
# 或在單次請求裡：{"chat_template_kwargs": {"enable_thinking": false}}
```

只看 tokens/s 的 benchmark 不會發現這件事；會檢查輸出的才會。

## 7. `n_ubatch` 512 對稀疏 MoE 是個糟糕的預設值

Prefill 是每個 ubatch 讀一次 expert。在 256 選 6 的路由之下，
一批 512 個 token 加起來幾乎會碰到每一個 expert，
所以**每個 ubatch 都會把某一層的整個 expert 集合拖過匯流排** ——
而這個成本是攤提在「這一批裡有多少 token」上面的。

在 DeepSeek-V4-Flash 上調高它（137 GB，expert 放在 DDR5）：

| ubatch | Prefill | 顯存 |
|---|---|---|
| 512（預設） | 151 tok/s | 9.1 GB |
| **8192** | **775 tok/s** | 12.5 GB |

**用 3.5 GB 換 5.1 倍。** 超過 8192 之後 prefill 就持平了，顯存卻繼續爬，
所以並不是越大越好 —— 目標是「大到能把你的 prompt 切成 2~3 批」。

兩個但書。512 token 以下完全沒有收益（兩邊都是一批），
所以短對話回合一點好處都沒有。
另外生成速度不受影響 —— 解碼是 batch 為 1 的，
而這也正是讓這件事能被乾淨量測的原因。

這只有在「expert 位於一條慢速連結的另一端」時才重要。
一個完整放在顯存裡的 dense 模型不會有對應的斷崖。

## 8. `--tensor-split` 修不好 MoE 的不均分配

當一張卡塞爆、另一張閒著時，直覺反應就是去調 `--tensor-split`。
在 expert 被 offload 的 MoE 模型上，它完全沒有作用：

```
--n-cpu-moe 35，兩張 32 GB 的卡
  GPU0  8.4 GB      GPU1  31.1 GB     ← 24 GB 閒置
--tensor-split 1,1 與 3,1 → 兩者都 OOM
```

兩個機制疊在一起的結果很糟。Layer split 是**連續的** ——
一張卡拿前段的一批層，另一張拿其餘。
而 `--n-cpu-moe N` 留在 GPU 側的 expert 位於**最後** N 層，同樣是連續的。
那一整段會完整落在同一張卡上，移動切點沒辦法把它拆開。

解法是 `-ot`，明確指名張量：

```bash
-ot "blk\.(3[0-5])\.ffn_.*_exps\.weight=CUDA0" \
-ot "blk\.(3[6-9]|4[0-2])\.ffn_.*_exps\.weight=CUDA1" \
--cpu-moe
```

**每個旗標一組 regex 範圍，而且 `-ot` 要放在 `--cpu-moe` 之前。**
有兩種會靜默失效的寫法：把 `=CPU` 的 pattern 寫在前面（它會把全部吃掉），
或用逗號串接多個 pattern（貪婪的 `.*` 會吞掉分隔符，結果什麼都沒匹配到）。
兩種都不會報錯 —— 辨識方法是看顯存停在「零 expert 在 GPU」的基準值，
同時生成速度維持在全 CPU 的數字。

買第二張卡之前值得知道：正確做完這件事，換到的是實測 +23% 的 prefill，
以及**完全無法證明的生成增益**；而單純調高 `n_ubatch` 就能換到 5.1 倍的 prefill。

## 9. 桌面環境的程序會佔住獨立顯卡

兩個各自獨立的元凶，在我們的機器上合計約 590 MiB：

- **Xorg / GNOME Shell** —— 把顯示輸出改接內顯就能解決
  （BIOS：`Primary Display = IGFX`、`iGPU Multi-Monitor = Enabled`）。
- **GNOME Remote Desktop** —— 即使顯示已經搬走，它**仍然**佔著約 130 MiB，
  因為它為了 NVENC 而載入 `libcuda` + `libnvidia-encode`。

```ini
# ~/.config/systemd/user/gnome-remote-desktop.service.d/igpu.conf
[Service]
Environment=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
Environment=__GLX_VENDOR_LIBRARY_NAME=mesa
```

任何聲稱已經搬走的東西都要驗證：

```bash
grep -cE 'libcuda|libnvidia-encode' /proc/<pid>/maps   # 要是 0
```

## 10. 短生成不構成一次 tok/s 量測

第 4 條的鏡像。長到足以走過解碼路徑的生成，不等於長到足以計時。

在掃描 DeepSeek-V4-Flash 的 expert 放置時，每個配置取樣 32 個生成 token：

| GPU 上的 expert 層數 | 生成 |
|---|---|
| 13 | 15.86 |
| 14 | 13.70 |
| 15 | 14.34 |
| 16 | 13.38 |

顯存更多、速度更慢、而且沒有順序 —— 一張逼著你去解釋的表，
而解釋很容易生出來（路由是逐 token 的，那吞吐量當然會隨模型生成的內容變動）。
在同一套硬體上用 **256** token 重複三次：

```
12.09   12.04   12.07      <- ±0.2%
```

散布來自取樣本身。在 32 token 之下，每次請求的固定成本和首 token 路徑
仍佔了整個計時視窗的可見比例，而 llama.cpp 印出來的是整段的平均值 ——
殘差跟你想量的效果一樣大。

代價不是那次白跑的量測。代價是**一張有雜訊的表看起來像一個發現**，
而你伸手去抓的那個解釋會顯得很有機制感、而且撐得過審閱 ——
[DeepSeek-V4-Flash 那份紀錄](../results/rtx5090-32gb-deepseek-v4-flash.zh-TW.md)
就帶著這個錯誤過了一天。Prefill 不會有同樣的問題，
它按定義就是在幾千個 token 上計時的。

在對任何小於 10% 的差異下結論之前，用 256 token、至少重複三次。

---

範圍說明：這份清單涵蓋的是**單張顯卡上會消耗或浪費顯存**的東西 ——
那正是 ctxprobe 要回答的問題。溢出到顯卡之外的代價另外量在
[spill-cost.zh-TW.md](spill-cost.zh-TW.md)。
多卡配置不在範圍內，`llama-fit-params` 處理得很好。

延伸閱讀：[model-quant.zh-TW.md](model-quant.zh-TW.md) 談怎麼挑一個裝得下的量化，
[kv-cache-quant.zh-TW.md](kv-cache-quant.zh-TW.md) 談怎麼把隨 context 成長的
cache 砍半。
