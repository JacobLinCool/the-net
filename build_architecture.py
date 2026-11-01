"""Build sparse safetensors shards from the OmniModel parameter graph."""

import argparse
import json
import math
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from build_checkpoint import build_tokenizer
from the_net.configuration_omni import OmniConfig
from the_net.modeling_omni import OmniModel


def component(name):
    prefix = name.split(".")[0]
    return {
        "world_input": "world_model",
        "world_detail_projection": "world_model",
        "text_head": "text_encoder",
        "memory_input": "memory_control",
        "memory_output": "memory_control",
    }.get(prefix, prefix)


def shard_group(name):
    parts = name.split(".")
    if "blocks" in parts:
        i = parts.index("blocks")
        return ".".join(parts[: i + 2])
    return component(name) + ".adapters"


def write_shard(path, entries):
    """Place small nonzero metadata first, then zero-filled sparse matrix payloads."""
    header = {
        "__metadata__": {
            "format": "pt",
            "initialization": "constant, untrained",
            "packed_encoding": "compressed-tensors int32, one bit",
        }
    }
    offset = 0
    payloads = []
    entries = sorted(entries, key=lambda x: (x[3] is None, x[0]))
    for name, dtype, shape, payload in entries:
        size = math.prod(shape) * {"F16": 2, "F32": 4, "I32": 4, "I64": 8}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        if payload is not None:
            if len(payload) != size:
                raise ValueError(f"Wrong byte count: {name}")
            payloads.append((offset, payload))
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        base = 8 + len(encoded)
        f.truncate(base + offset)
        for start, payload in payloads:
            f.seek(base + start)
            f.write(payload)
    with safe_open(path, framework="pt") as f:
        if set(f.keys()) != {e[0] for e in entries}:
            raise ValueError("Safetensors key mismatch")
        for name, _, shape, _ in entries:
            if f.get_slice(name).get_shape() != shape:
                raise ValueError(f"Safetensors shape mismatch: {name}")
    return offset


def build(destination, tiny=False):
    destination.mkdir(parents=True, exist_ok=False)
    config = OmniConfig.tiny() if tiny else OmniConfig()
    with torch.device("meta"):
        model = OmniModel(config)
    modules = dict(model.named_modules())
    groups = defaultdict(list)
    logical = {}
    totals = defaultdict(int)
    active = defaultdict(int)
    for name, parameter in model.named_parameters():
        shape = list(parameter.shape)
        count = parameter.numel()
        totals[component(name)] += count
        multiplier = 1.0
        if ".experts." in name:
            parent = name.split(".experts.")[0]
            mlp = modules[parent]
            multiplier = mlp.selected / len(mlp.experts)
        active[component(name)] += int(count * multiplier)
        module_name, leaf = name.rsplit(".", 1)
        module = modules[module_name]
        target = groups[shard_group(name)]
        if isinstance(module, nn.Linear) and leaf == "weight":
            rows, columns = shape
            packed_name = module_name + ".weight_packed"
            packed_shape = [rows, math.ceil(columns / 32)]
            target.extend(
                [
                    (packed_name, "I32", packed_shape, None),
                    (module_name + ".weight_shape", "I64", [2], struct.pack("<qq", rows, columns)),
                    (
                        module_name + ".weight_scale",
                        "F32",
                        [1],
                        struct.pack("<f", 1 / math.sqrt(columns)),
                    ),
                ]
            )
            logical[name] = {
                "shape": shape,
                "stored_key": packed_name,
                "stored_shape": packed_shape,
                "encoding": "int32-packed-1bit",
                "parameters": count,
            }
        else:
            payload = None
            if isinstance(module, nn.RMSNorm):
                payload = struct.pack("<e", 1.0) * count
            target.append((name, "F16", shape, payload))
            logical[name] = {
                "shape": shape,
                "stored_key": name,
                "stored_shape": shape,
                "encoding": "float16",
                "parameters": count,
            }
    index = {}
    payload_bytes = 0
    for number, (group, entries) in enumerate(sorted(groups.items())):
        relative = f"weights/{group}.safetensors"
        payload_bytes += write_shard(destination / relative, entries)
        for name, *_ in entries:
            if name in index:
                raise ValueError(f"Duplicate tensor name: {name}")
            index[name] = relative
        if (number + 1) % 100 == 0:
            print(f"Validated {number + 1}/{len(groups)} sparse shards", flush=True)
    total = sum(totals.values())
    if not tiny and not 20_000_000_000_000 <= total <= 30_000_000_000_000:
        raise ValueError("Architecture is outside the 20T–30T interval")
    config.architectures = ["OmniModel"]
    config.auto_map = {
        "AutoConfig": "configuration_omni.OmniConfig",
        "AutoModel": "modeling_omni.OmniModel",
    }
    config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "linear": {
                "targets": ["Linear"],
                "format": "pack-quantized",
                "weights": {
                    "num_bits": 1,
                    "type": "int",
                    "strategy": "tensor",
                    "symmetric": True,
                    "dynamic": False,
                },
            }
        },
    }
    config.save_pretrained(destination)

    def write_json(name, value):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")

    write_json(
        "model.safetensors.index.json",
        {"metadata": {"total_size": payload_bytes, "total_parameters": total}, "weight_map": index},
    )
    write_json("logical_tensor_shapes.json", logical)
    root = Path(__file__).parent
    for name in ["configuration_omni.py", "modeling_omni.py", "packed_checkpoint.py"]:
        shutil.copyfile(root / "the_net" / name, destination / name)
    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    shutil.copyfile(root / "OMNI_MODEL_CARD.md", destination / "README.md")
    shutil.copyfile(root / "requirements-model.txt", destination / "requirements.txt")
    build_tokenizer().save_pretrained(destination)
    report = {
        "logical_parameters": total,
        "components": dict(sorted(totals.items())),
        "one_route_parameter_slots_by_component": dict(sorted(active.items())),
        "logical_weight_tensors": len(logical),
        "stored_tensors": len(index),
        "shards": len(groups),
        "payload_bytes": payload_bytes,
        "storage_limit_basis": "local st_blocks * 512; backend allocation unverified",
        "backend_physical_storage_bytes": None,
        "initialization": "Packed linear weights decode to -1/sqrt(fan_in); embeddings and convolutions are zero; RMSNorm weights are one. Untrained, not a random pretrained checkpoint.",
    }
    write_json("architecture_report.json", report)
    allocated = sum(p.stat().st_blocks * 512 for p in destination.rglob("*") if p.is_file())
    report["allocated_file_bytes"] = allocated
    write_json("architecture_report.json", report)
    if allocated >= 500_000_000:
        raise ValueError(f"Allocated file data exceeds 500 MB: {allocated}")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni-20t"))
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    build(args.output, args.tiny)
