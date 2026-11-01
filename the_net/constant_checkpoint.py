"""Constant-tensor checkpoint reader."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .configuration_omni import OmniConfig
from .modeling_omni import OmniModel


class ConstantCheckpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.shapes = json.loads((self.directory / "tensor_shapes.json").read_text())
        self.values = load_file(self.directory / "model.safetensors")
        if self.shapes.keys() != self.values.keys():
            raise ValueError("Shape and value keys differ")
        self.config = OmniConfig.from_pretrained(self.directory, local_files_only=True)

    def slice(self, name, start, length):
        size = 1
        for dimension in self.shapes[name]:
            size *= dimension
        if (
            type(start) is not int
            or type(length) is not int
            or start < 0
            or length < 0
            or start + length > size
        ):
            raise ValueError("Invalid tensor slice")
        if length > 1_000_000:
            raise ValueError("Read at most one million elements per slice")
        return self.values[name].expand(length).clone()

    def meta_model(self):
        with torch.device("meta"):
            model = OmniModel(self.config)
        parameters = dict(model.named_parameters())
        if parameters.keys() != self.shapes.keys():
            raise ValueError("Checkpoint and model parameter keys differ")
        for name, parameter in parameters.items():
            if list(parameter.shape) != self.shapes[name]:
                raise ValueError(f"Shape mismatch: {name}")
        return model

    def materialize(self, max_parameters=10_000_000):
        model = self.meta_model()
        total = sum(p.numel() for p in model.parameters())
        if total > max_parameters:
            raise ValueError(f"{total:,} parameters exceed the materialization limit")
        model.to_empty(device="cpu")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.fill_(self.values[name].item())
        return model
