# The Net

異質階層全模態模型架構：文字、音訊、影像、影片與 EEG encoder，搭配
多尺度融合、Hierarchical Sparse MoE、媒體 decoder 與記憶模組。

- **20T 級**邏輯架構
- **686** 個 attention／FFN blocks
- **68,464** 個具名權重張量
- 共享核心寬度 **8,192 / 12,288 / 16,384**，每層 **128 experts，top-2 routing**
- Checkpoint：常數初始化、共享儲存

11 個主要 component 的架構介紹見 [模型卡](OMNI_MODEL_CARD.md)。

## Compact checkpoint

目前的常數權重以「每個張量的數值＋原始 shape」儲存，可精確還原初始化值。
`tensor_shapes.json` 保留完整層級名稱與矩陣維度；safetensors 儲存對應數值。
Hub 的 tensor 查看器顯示實際儲存形狀。

```bash
.venv/bin/python build_compact.py --output artifacts/the_net_compact
```

```python
from the_net.constant_checkpoint import ConstantCheckpoint

checkpoint = ConstantCheckpoint("artifacts/the_net_compact")
model = checkpoint.meta_model()
print(sum(p.numel() for p in model.parameters()))
name = "world_model.stages.2.blocks.0.mlp.experts.0.gate_proj.weight"
print(checkpoint.shapes[name])
print(checkpoint.slice(name, 0, 16))
```

## Small runtime

同一組模組類別提供 1.899M 參數的小型設定，可在 CPU 上執行。

```bash
.venv/bin/python build_checkpoint.py --output artifacts/omni-runtime
```

載入方式與輸入格式見 [runtime 模型卡](RUNTIME_CARD.md)。

## 開發

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-model.txt pytest ruff
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

測試包含精確參數計數、五模態梯度、AutoClass round-trip、shape 一致性、
compact checkpoint 的常數值還原，以及 shared-bank 別名與 bit 讀取。

## 發布狀態

[Hugging Face 原生查看器](https://huggingface.co/JacobLinCool/the-net?show_file_info=model.safetensors.index.json) 已確認顯示 **20.2T params** 與階層式 tensor 清單。
一份 `architecture.safetensors` 保留 68,464 個架構名稱；9,400 個分片路徑共用一份 U8 bank。
Hub 回報 usedStorage 為 392,572,666 bytes；全部不重複檔案合計 405,144,382 bytes。
原生查看器顯示儲存區塊 shape；完整架構 shape 位於 `tensor_shapes.json`。
原生計數包含重複共享的 bit slots，與獨立學習權重數不同。

`SharedBankCheckpoint` 明確解析 `the-net-shared-aliases-v2` 的別名索引，提供有界的儲存區塊與 bit 讀取。
建立封裝：`.venv/bin/python build_shared_display.py`。
發布工具 `publish_shared_display.py --recreate` 會刪除並重建遠端 repository，檢查不重複 LFS 儲存量，再用伺服器端複製建立分片路徑。

完整 metadata 共 9,401 個 headers、77,864 個 tensor entries，原始 headers 合計約 10.73 MB。
本次完整 API 回應為 11,537,231 bytes，下載及解析耗時 29.222 秒。
最新驗證：`artifacts/shared-display-v2-verification.json`；原生 badge 與完整索引清單均已在瀏覽器確認。
