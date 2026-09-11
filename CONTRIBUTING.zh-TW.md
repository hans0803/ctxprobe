# 貢獻指南

[English](CONTRIBUTING.md) · **繁體中文**

最有用的貢獻是**一張我們沒有的卡的實測**。[results/](results/) 裡的每一個數字
都來自同一個人的兩台機器 —— 就跟你想的一樣窄。

## 送一張卡的數據

```bash
./ctxprobe ~/models/YourModel-Q4_K_M.gguf --json > result.json
```

開一個 issue，把那個檔案貼進來，另外附上：

- **確切的模型檔** —— Hugging Face 的 repo id 加檔名，不是「Qwen 27B Q4」。
  兩個掛著同一個量化名字的檔案，每權重可以差半個 bit；
  見 [model-quant.zh-TW.md](docs/model-quant.zh-TW.md)。
- **卡上還有什麼** —— 桌面環境、其他程序。執行時會印出它找到的，
  但它不可能知道之後會有什麼跑起來。
- **llama.cpp 的 commit。** 天花板會隨版本移動。

這樣就夠了。你不需要寫一頁文件。

### 什麼樣的結果會難以使用

- **`validated_at_fill: false`。** 階梯放行了那個大小，但滿視窗撐不住，
  而且搜尋在它底下也找不到撐得住的。那個數字不是天花板，log 才是有價值的東西。
  用 `--keep` 重跑一次並附上 log。
- **`--ngl` 低於層數。** 那樣模型有一部分在系統記憶體裡，天花板就不是卡的性質了。
  刻意量溢出是歡迎的 —— 講清楚就好，見 [spill-cost.zh-TW.md](docs/spill-cost.zh-TW.md)。
- **`--parallel` 大於 1** 卻沒說明。KV 是按 slot 配置的，那個數字的意義不同。

## JSON 格式

`--json` 會往 stdout 寫一個物件，進度輸出到 stderr。Schema 4：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `schema` | int | 形狀改變時會遞增。目前是 **4**。 |
| `model` | string | 傳入的 GGUF 的檔名。 |
| `gpu` | string | `nvidia-smi` 給的名字。 |
| `max_context` | int | **答案。** 實際跑得起來的最大 context。 |
| `validated_at_fill` | bool | `max_context` 有沒有撐過一個灌滿視窗 `fill_percent` 的 prompt。**如果是 false，`max_context` 就是未確認的** —— 階梯放行了它，而滿視窗沒有。 |
| `ladder_max_context` | int | 光靠 64/512/4096 階梯會回報的數字。除非滿視窗失敗、搜尋往下二分過，否則等於 `max_context` —— 那個落差會很大的唯一一種架構，見 [gotchas.zh-TW.md](docs/gotchas.zh-TW.md) 的 #11。 |
| `fill_timed_out` | bool | 滿視窗的 prompt 超過了時間預算，而伺服器仍然健康。此時 `max_context` 是因為速度而未確認，不是記憶體 —— 調高 `-ub` 或 `CTXPROBE_FILL_MIN_TPS` 再跑一次。 |
| `peak_vram_mib` | int\|null | 勝出那一輪在被探測的裝置上取樣到的最高用量。這**不是**餘裕的估計 —— [gotchas.zh-TW.md](docs/gotchas.zh-TW.md) #5。 |
| `prefill_tps` | float\|null | 來自填充驗證，所以是對著已載入的視窗量的。 |
| `generate_tps` | float\|null | 同一輪。是往接近滿的快取裡解碼，不是空的。 |
| `kv_cache_type` | string | `--kv`。 |
| `slots` | int | `--parallel`。KV 是每個 slot 各配置一份。 |
| `n_gpu_layers` | int | `--ngl`。低於模型層數就代表溢出了。 |
| `fill_percent` | int | `--fill`，預設 95。 |
| `cuda_module_loading` | string | `LAZY (default)` 或 `EAGER`。兩者的天花板不同，見 README。 |
| `flash_attn` | string | 伺服器實際用的 —— 量化的 V cache 會把它釘成開啟，不管 `-fa` 寫什麼。 |
| `vram_total_mib` | int | `nvidia-smi` 回報的。 |
| `vram_allocatable_mib` | int\|null | `cudaMemGetInfo` 回報的，比較小。沒有可 import 的 `torch` 時是 null。 |
| `vram_held_by_others_mib` | int | 這一輪開始前就已經在使用的裝置記憶體。 |

每次啟動的資料寫在 `--out` 底下的 `results.tsv`，一行一次：

```
context  result  peak_vram_mib  prefill_tps  generate_tps  prompt_tokens
```

`result` 欄位以 `final:` 為前綴的是填充驗證。其餘是階梯探測，
它們的速度來自那一輪爬到的最後一階 —— 所以描述的是輕載的視窗，
**不能**跟 `final:` 的數字相比。

`result` 的值是 `PASS`、`LOAD_FAIL`、`DECODE_OOM`、`PREFILL_OOM`、`SILENT`、`TIMEOUT` 之一。
當某一階殺死了那一輪時，`prompt_tokens` 會帶著 `died@N`，指出是哪一階；
N 個 token 的 prompt 沒能在 S 秒預算內跑完時，則是 `timeout@N/Ss`。

## 修改腳本

它是一個 bash 檔，`set -uo pipefail`，除了 `llama-server` 和 `python3` 之外沒有相依。
CI 會跑 `shellcheck -S warning`、確認 `--help` 不會爆、
並確認四種壞參數是被擋掉而不是崩潰。

同一台機器上要同時跑多個，`--port` 必須不同。腳本會對每個 port 取一個原子鎖，
因為啟動時的 port 檢查只是一個時間點，兩個 run 可以同時通過 ——
之後輸的那個伺服器綁不上 port，但它的請求會打到贏的那個身上，
於是它會為一個從未跑起來的配置回報 PASS。

動手之前還有兩件事值得知道：

- **`-e` 是刻意關掉的。** 探測本來就會失敗，那就是量測本身。
  要檢查結束碼的地方請明確地檢查。
- **回報結果，不是回報呼叫。** `curl` 回傳 0 並沒有告訴你伺服器好了 ——
  它對 HTTP 503 也回 0，而且連線中途斷掉時它會留下一個過期的 body。
  這個腳本裡「就緒」的定義是一個真的生成回傳 200。
  [rtx5090-32gb-qwen3.8-flash-next.zh-TW.md](results/rtx5090-32gb-qwen3.8-flash-next.zh-TW.md)
  有好幾節的存在，就是因為這條規則被違反過。

## 新增一頁實測

只有你想寫才需要。標準是**頁面上的每一個數字都能從頁面上的東西重現** ——
確切的參數、確切的檔案、機器。
如果某個數字來自機制推導而不是量測，講明是哪一種。
如果後來的實測推翻了先前的頁面，先前那頁要加更正，而不是靜靜改掉；
`results/rtx5060ti-16gb-qwen3.6-27b.zh-TW.md` 裡有一個例子。

頁面是雙語的（`*.md` 和 `*.zh-TW.md`），而且兩邊的數字必須一致。
只送英文版是可以的 —— 講一聲，可以再翻。
