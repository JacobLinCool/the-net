"""Serialize the constant-initialized OmniModel as values plus tensor shapes."""

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from build_checkpoint import build_tokenizer
from the_net.configuration_omni import OmniConfig
from the_net.modeling_omni import OmniModel


def build(destination, tiny=False):
    destination.mkdir(parents=True, exist_ok=False)
    config = OmniConfig.tiny() if tiny else OmniConfig()
    config.checkpoint_format = "constant-tensors-v1"
    config.initialization = "constant, untrained"
    config.architectures = ["OmniModel"]
    config.auto_map = {
        "AutoConfig": "configuration_omni.OmniConfig",
        "AutoModel": "modeling_omni.OmniModel",
    }
    config.save_pretrained(destination)
    with torch.device("meta"):
        model = OmniModel(config)
    modules = dict(model.named_modules())
    values, shapes = {}, {}
    total = 0
    for name, parameter in model.named_parameters():
        parent, leaf = name.rsplit(".", 1)
        module = modules[parent]
        value = 0.0
        if isinstance(module, nn.Linear) and leaf == "weight":
            # Match the float32 scale stored by the packed checkpoint writer.
            value = -struct.unpack("<f", struct.pack("<f", 1 / math.sqrt(parameter.shape[1])))[0]
        elif isinstance(module, nn.RMSNorm):
            value = 1.0
        values[name] = torch.tensor([value], dtype=torch.float32, device="cpu")
        shapes[name] = list(parameter.shape)
        total += parameter.numel()
    save_file(
        values,
        destination / "model.safetensors",
        metadata={
            "format": "constant-tensors-v1",
            "logical_parameters": str(total),
            "reconstruction": "Each logical tensor is filled with its named float32 scalar",
        },
    )
    (destination / "tensor_shapes.json").write_text(
        json.dumps(shapes, separators=(",", ":")) + "\n"
    )
    root = Path(__file__).parent
    for name in (
        "configuration_omni.py",
        "modeling_omni.py",
        "constant_checkpoint.py",
        "__init__.py",
    ):
        shutil.copyfile(root / "the_net" / name, destination / name)
    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    shutil.copyfile(root / "OMNI_MODEL_CARD.md", destination / "README.md")
    shutil.copyfile(root / "requirements-model.txt", destination / "requirements.txt")
    build_tokenizer().save_pretrained(destination)
    report = {
        "format": "constant-tensors-v1",
        "logical_parameters": total,
        "logical_tensors": len(shapes),
        "stored_scalars": len(values),
        "initialization": "constant, untrained",
    }
    report_path = destination / "checkpoint_report.json"
    for _ in range(3):
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["file_bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if report["file_bytes"] >= 500_000_000:
        raise ValueError("Compact checkpoint exceeds 500 MB")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/the_net_compact"))
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    build(args.output, args.tiny)
