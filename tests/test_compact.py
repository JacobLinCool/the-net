import math
import struct

import pytest
import torch
from torch import nn

from build_compact import build
from the_net.constant_checkpoint import ConstantCheckpoint


def test_compact_restores_constant_initialization(tmp_path):
    report = build(tmp_path / "compact", tiny=True)
    model = ConstantCheckpoint(tmp_path / "compact").materialize().eval()
    assert report["file_bytes"] < 500_000_000
    assert sum(p.numel() for p in model.parameters()) == report["logical_parameters"]
    modules = dict(model.named_modules())
    for name, parameter in model.named_parameters():
        parent, leaf = name.rsplit(".", 1)
        module = modules[parent]
        expected = 0.0
        if isinstance(module, nn.Linear) and leaf == "weight":
            scale = 1 / math.sqrt(parameter.shape[1])
            expected = -struct.unpack("<f", struct.pack("<f", scale))[0]
        elif isinstance(module, nn.RMSNorm):
            expected = 1.0
        assert torch.equal(parameter, torch.full_like(parameter, expected)), name
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[1, 4, 5]])).next_token_logits).all()
    assert model.generate(torch.tensor([[1, 4]]), max_new_tokens=2).shape == (1, 4)


def test_compact_slices_and_allocation_limit(tmp_path):
    build(tmp_path / "compact", tiny=True)
    checkpoint = ConstantCheckpoint(tmp_path / "compact")
    name = "world_model.stages.2.blocks.0.mlp.experts.0.gate_proj.weight"
    assert checkpoint.slice(name, 0, 16).shape == (16,)
    with pytest.raises(ValueError):
        checkpoint.slice(name, -1, 16)
    with pytest.raises(ValueError):
        checkpoint.materialize(max_parameters=1)
