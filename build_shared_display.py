"""Build a bounded shared-bank checkpoint with one architecture header."""

import json
import math
import shutil
import struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

SOURCE = Path("artifacts/the_net_compact")
DESTINATION = Path("artifacts/shared-display-v2")
COPIES = 9400
BANK_SHAPE = [16384, 16384]


def write_file(path, entries, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, value in entries:
        size = math.prod(shape) * {"F16": 2, "U8": 1}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for _, dtype, shape, value in entries:
            unit = struct.pack("<e", value) if dtype == "F16" else bytes([value])
            remaining = math.prod(shape)
            block = unit * min(remaining, 1024 * 1024)
            while remaining:
                count = min(remaining, len(block) // len(unit))
                stream.write(block[: count * len(unit)])
                remaining -= count
    with safe_open(path, framework="pt") as f:
        assert set(f.keys()) == {e[0] for e in entries}
        for name, _, shape, _ in entries:
            assert f.get_slice(name).get_shape() == shape
    return offset, len(encoded)


def build(destination=DESTINATION):
    destination.mkdir(parents=True, exist_ok=False)
    shapes = json.loads((SOURCE / "tensor_shapes.json").read_text())
    values = load_file(SOURCE / "model.safetensors")
    entries = []
    for name, shape in sorted(shapes.items()):
        stored = (
            shape
            if len(shape) <= 1
            else [math.ceil(shape[0] / 512), math.ceil(math.prod(shape[1:]) / 1024)]
        )
        entries.append((name, "F16", stored, values[name].item()))
    architecture_bytes, architecture_header = write_file(
        destination / "architecture.safetensors",
        entries,
        {
            "format": "pt",
            "shape_semantics": "Stored architecture blocks; full shapes in tensor_shapes.json",
        },
    )
    paths = [
        f"shards/{i // 512:03d}/model-{i + 1:05d}-of-{COPIES:05d}.safetensors"
        for i in range(COPIES)
    ]
    bank_bytes, bank_header = write_file(
        destination / paths[0],
        [("shared_bank.weight", "U8", BANK_SHAPE, 0)],
        {"format": "pt", "encoding": "unsigned one-bit shared storage bank"},
    )
    expected = architecture_bytes // 2 + bank_bytes * COPIES * 8
    metadata = {
        "format": "the-net-shared-aliases-v2",
        "alias_prefix": "shared_bank.views.",
        "alias_tensor": "shared_bank.weight",
        "canonical_bank": paths[0],
        "total_size": architecture_bytes + bank_bytes * COPIES,
        "total_parameters": expected,
    }
    weight_map = {name: "architecture.safetensors" for name in shapes}
    weight_map.update({f"shared_bank.views.{i:05d}.weight": p for i, p in enumerate(paths)})
    index = {"metadata": metadata, "weight_map": weight_map}
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(index, separators=(",", ":")) + "\n"
    )
    config = json.loads((SOURCE / "config.json").read_text())
    config["checkpoint_format"] = metadata["format"]
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "shared_bank": {
                "targets": ["shared_bank"],
                "weights": {
                    "num_bits": 1,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "tensor",
                    "dynamic": False,
                },
            }
        },
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    names = [
        "LICENSE",
        "__init__.py",
        "configuration_omni.py",
        "modeling_omni.py",
        "requirements.txt",
        "tensor_shapes.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for name in names:
        shutil.copyfile(SOURCE / name, destination / name)
    shutil.copyfile(SOURCE / "model.safetensors", destination / "initialization.st")
    reader = (
        (SOURCE / "constant_checkpoint.py")
        .read_text()
        .replace('"model.safetensors"', '"initialization.st"')
    )
    (destination / "constant_checkpoint.py").write_text(reader)
    shutil.copyfile("the_net/shared_checkpoint.py", destination / "shared_checkpoint.py")
    shutil.copyfile("OMNI_MODEL_CARD.md", destination / "README.md")
    shutil.copyfile("EXPERIMENT.md", destination / "EXPERIMENT.md")
    (destination / ".gitattributes").write_text(
        "* text=auto\n*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.st filter=lfs diff=lfs merge=lfs -text\n"
    )
    report = {
        "format": metadata["format"],
        "architecture_tensors": len(shapes),
        "bank_aliases": COPIES,
        "expanded_header_entries": len(shapes) + COPIES,
        "expanded_raw_header_bytes": architecture_header + bank_header * COPIES,
        "expected_native_count": expected,
        "logical_architecture_parameters": sum(math.prod(s) for s in shapes.values()),
        "count_semantics": "Architecture blocks plus repeated shared-bank bit slots",
    }
    report_path = destination / "shared_storage_report.json"
    for _ in range(3):
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["unique_file_bytes"] = sum(
            p.stat().st_size for p in destination.rglob("*") if p.is_file()
        )
    assert report["unique_file_bytes"] < 450_000_000
    assert report["expanded_raw_header_bytes"] < 20_000_000
    assert 20_000_000_000_000 <= expected <= 30_000_000_000_000
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    build()
