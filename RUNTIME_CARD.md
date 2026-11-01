---
library_name: transformers
license: mit
tags:
- custom_code
- multimodal
---

# The Net — Small

A compact configuration of the The Net architecture with five input modalities,
hierarchical chunking, sparse experts, media heads and memory.

| Setting | Value |
| --- | --- |
| Parameters | 1,899,346 |
| Widths | 32 / 64 / 96 |
| Blocks per stage | 1 |
| Experts per block | Up to 2 |
| Checkpoint | Random initialization, seed 2030; untrained |
| File size | Approximately 7.67 MB |

```python
from transformers import AutoModel, AutoTokenizer

path = "artifacts/omni-runtime"
model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path)
ids = tokenizer("你好，世界", return_tensors="pt")["input_ids"]
output = model.generate(ids, max_new_tokens=8)
```

Use the local runtime built with `build_checkpoint.py`. The interface accepts one
example per call and returns next-token logits and memory state. Optional outputs
include audio codebook logits, image latents and video latents.

| Input | Shape |
| --- | --- |
| Text | 1 × sequence |
| RGB image | 1 × 3 × H × W; H/W multiples of 4 |
| Audio | 1 × T; T multiple of 32 |
| RGB video | 1 × 3 × F × H × W; F even, H/W multiples of 4 |
| EEG | 1 × 8 × T, plus an 8×8 electrode adjacency matrix |
| Task vector / memory | 1 × 32 |

Set `output_modalities=("audio", "image", "video")` to select media heads.
