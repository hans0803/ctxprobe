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

## 3. `n_ctx` 會被對齊到 256

```c
// src/llama-context.cpp
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
```

`-c 35000` 會變成 35072。256 才是真正的搜尋粒度；
用 1024 當步進每一階會跳過三個可測的尺寸。可以從 log 的 `n_ctx_slot` 確認。

## 4. 短 prompt 無法認證一個 context 大小

這是最大的一個。prefill 的 compute buffer 會隨 prompt 長度成長，於是會發生這種事：

| Context | 載入 | 短 prompt | 真實長度的 prompt |
|---|---|---|---|
| 34816 | 成功 | 25.99 tok/s | 正常 |
| 35072 | 成功 | 25.98 tok/s | **CUDA OOM** |

載入過程、健康檢查、小量生成 —— 沒有任何一項能區分這兩者。
只有「灌滿視窗的 prompt」可以。

相關的一點：子程序這樣死掉之後會變成殭屍程序（defunct），
而只追蹤自己狀態的上層 gateway 會繼續回報這個部署是健康的。
要檢查程序，不是檢查狀態端點。

## 5. Thinking 模型會回傳空的 content

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

## 6. 桌面環境的程序會佔住獨立顯卡

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

---

範圍說明：這份清單涵蓋的是**單張顯卡上會消耗或浪費顯存**的東西 ——
那正是 ctxprobe 要回答的問題。溢出到顯卡之外的代價另外量在
[spill-cost.zh-TW.md](spill-cost.zh-TW.md)。
多卡配置不在範圍內，`llama-fit-params` 處理得很好。

延伸閱讀：[model-quant.zh-TW.md](model-quant.zh-TW.md) 談怎麼挑一個裝得下的量化，
[kv-cache-quant.zh-TW.md](kv-cache-quant.zh-TW.md) 談怎麼把隨 context 成長的
cache 砍半。
