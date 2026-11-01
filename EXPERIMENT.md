# The Net: release notes

This release is a **Safetensors & Xet storage experiment**. It publishes the complete The Net
architecture, with every tensor name and dimension, inside a few hundred megabytes of unique
storage. The weights are constant-initialized and shared; this is not a trained 20T checkpoint,
and the model produces no meaningful output.

## Weights

Every architecture tensor holds a single constant. Linear weights are −1/√fan_in rounded to
float32, RMSNorm weights are one and all other tensors are zero. `initialization.st` stores one
scalar per tensor, and `tensor_shapes.json` gives the full shape it expands to.

## Storage layout

| File | Contents |
| --- | --- |
| `architecture.safetensors` | All 68,464 architecture tensors, stored as reduced F16 blocks |
| `shards/**.safetensors` | 9,400 paths that share one U8 storage bank |
| `initialization.st` | Constant value of each architecture tensor |
| `tensor_shapes.json` | Full architecture shapes |
| `model.safetensors.index.json` | Index in the `the-net-shared-aliases-v2` format |

The shard paths point to the same Xet object, so the bank is stored once however many paths
reference it. The index is not a generic Transformers sharded checkpoint: it marks the bank
aliases explicitly, and `shared_checkpoint.py` resolves them.

## Parameter counts

The architecture defines 20,088,058,521,618 parameters, derived from the modules in
`modeling_omni.py`. The Hub's native counter reports about 20.2T because it counts the stored
blocks plus every repeated bit slot of the shared bank. That figure reflects storage aliasing,
not independently learned weights.

## Reading the checkpoint

```python
import json

from shared_checkpoint import SharedBankCheckpoint

checkpoint = SharedBankCheckpoint(".")
name = "world_model.stages.2.blocks.0.mlp.experts.0.gate_proj.weight"
print(checkpoint.tensor_info(name))  # stored block
print(checkpoint.stored_slice(name, 0, 4))
print(json.load(open("tensor_shapes.json"))[name])  # full shape
print(checkpoint.bit_slice("shared_bank.views.00042.weight", 0, 8))
```

Reads are bounded to one million elements per call. The full architecture is never materialized;
the same classes run end to end at a small configuration.
