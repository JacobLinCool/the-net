"""One-bit packed checkpoint loading."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .configuration_omni import OmniConfig
from .modeling_omni import OmniModel


def unpack_one_bit(packed, shape):
    rows, columns = shape
    if packed.dtype != torch.int32 or tuple(packed.shape) != (rows, (columns + 31) // 32):
        raise ValueError("Invalid packed matrix")
    bits = torch.arange(32, device=packed.device)
    unsigned = (packed[..., None] >> bits) & 1
    return unsigned.reshape(rows, -1)[:, :columns].to(torch.float32) - 1


def load_packed_checkpoint(directory, max_parameters=10_000_000):
    directory = Path(directory)
    manifest = json.loads((directory / "logical_tensor_shapes.json").read_text())
    total = sum(item["parameters"] for item in manifest.values())
    if total > max_parameters:
        raise ValueError(
            f"Materializing {total:,} parameters exceeds the explicit allocation limit"
        )
    config = OmniConfig.from_pretrained(directory, local_files_only=True)
    model = OmniModel(config)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            item = manifest[name]
            key = item["stored_key"]
            with safe_open(directory / index[key], framework="pt") as f:
                stored = f.get_tensor(key)
                if item["encoding"] == "int32-packed-1bit":
                    prefix = key.removesuffix("weight_packed")
                    shape = f.get_tensor(prefix + "weight_shape").tolist()
                    if shape != item["shape"]:
                        raise ValueError(f"Original shape mismatch: {name}")
                    stored = unpack_one_bit(stored, shape) * f.get_tensor(prefix + "weight_scale")
                if list(stored.shape) != list(parameter.shape):
                    raise ValueError(f"Model shape mismatch: {name}")
                parameter.copy_(stored)
    return model
