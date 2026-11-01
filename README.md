# The Net

A heterogeneous, hierarchical omni-modal architecture: text, audio, image, video and EEG encoders,
combined with multiscale fusion, a hierarchical sparse MoE, media decoders and a memory module.

- **20T-scale** logical architecture
- **686** attention/FFN blocks
- **68,464** named weight tensors
- Shared core widths **8,192 / 12,288 / 16,384**, with **128 experts per layer and top-2 routing**
- Checkpoint: constant initialization, shared storage

See the [model card](OMNI_MODEL_CARD.md) for an overview of the 11 major components.

## Compact checkpoint

The constant weights are stored as one value plus the original shape per tensor, which restores
the initialization exactly. `tensor_shapes.json` keeps the full hierarchical names and matrix
dimensions; the safetensors file stores the matching values. The Hub's tensor viewer shows the
stored shapes.

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

The same module classes provide a small 1.899M-parameter configuration that runs on CPU.

```bash
.venv/bin/python build_checkpoint.py --output artifacts/omni-runtime
```

See the [runtime model card](RUNTIME_CARD.md) for loading and input formats.

## Development

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-model.txt pytest ruff
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The tests cover exact parameter counts, gradients across all five modalities, AutoClass
round-trips, shape consistency, restoration of the compact checkpoint's constant values, and
shared-bank alias and bit reads.

## Release status

The [Hugging Face native viewer](https://huggingface.co/JacobLinCool/the-net?show_file_info=model.safetensors.index.json)
is confirmed to show **20.2T params** and the hierarchical tensor list. A single
`architecture.safetensors` holds the 68,464 architecture names; 9,400 shard paths share one U8
bank. The Hub reports a usedStorage of 392,572,666 bytes; all unique files total 405,144,382 bytes.
The native viewer shows stored block shapes; the full architecture shapes are in
`tensor_shapes.json`. The native count includes repeated shared bit slots and differs from the
number of independently learned weights. Release notes: [EXPERIMENT.md](EXPERIMENT.md).

`SharedBankCheckpoint` explicitly resolves the `the-net-shared-aliases-v2` alias index and
provides bounded reads of stored blocks and bits. Build the package with
`.venv/bin/python build_shared_display.py`. The publisher `publish_shared_display.py --recreate`
deletes and recreates the remote repository, checks unique LFS storage, then creates the shard
paths with server-side copies.

The full metadata spans 9,401 headers and 77,864 tensor entries, about 10.73 MB of raw headers.
The latest full API response was 11,537,231 bytes and took 29.222 seconds to download and parse.
Latest verification: `artifacts/shared-display-v2-verification.json`; the native badge and the
full index listing were both confirmed in the browser.
