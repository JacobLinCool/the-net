import torch

from build_architecture import build as build_sparse
from build_compact import build
from the_net.constant_checkpoint import ConstantCheckpoint
from the_net.packed_checkpoint import load_packed_checkpoint


def test_compact_exactly_matches_packed_checkpoint(tmp_path):
    build_sparse(tmp_path / "sparse", tiny=True)
    report = build(tmp_path / "compact", tiny=True)
    original = load_packed_checkpoint(tmp_path / "sparse").eval()
    checkpoint = ConstantCheckpoint(tmp_path / "compact")
    restored = checkpoint.materialize().eval()
    assert report["file_bytes"] < 500_000_000
    assert original.state_dict().keys() == restored.state_dict().keys()
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(
            original(ids).next_token_logits, restored(ids).next_token_logits, rtol=0, atol=0
        )


def test_compact_slices_and_allocation_limit(tmp_path):
    import pytest

    build(tmp_path / "compact", tiny=True)
    checkpoint = ConstantCheckpoint(tmp_path / "compact")
    name = "world_model.stages.2.blocks.0.mlp.experts.0.gate_proj.weight"
    assert checkpoint.slice(name, 0, 16).shape == (16,)
    with pytest.raises(ValueError):
        checkpoint.slice(name, -1, 16)
    with pytest.raises(ValueError):
        checkpoint.materialize(max_parameters=1)
