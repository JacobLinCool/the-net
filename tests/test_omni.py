import importlib.util
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from build_architecture import build
from the_net.configuration_omni import OmniConfig
from the_net.modeling_omni import OmniModel
from the_net.packed_checkpoint import load_packed_checkpoint, unpack_one_bit


def test_tiny_graph_multimodal_and_gradients():
    torch.manual_seed(2030)
    model = OmniModel(OmniConfig.tiny())
    output = model(
        torch.tensor([[1, 10, 11, 12]]),
        pixel_values=torch.randn(1, 3, 8, 8),
        audio_values=torch.randn(1, 64),
        video_values=torch.randn(1, 3, 2, 8, 8),
        eeg_values=torch.randn(1, 8, 8),
        eeg_adjacency=torch.eye(8),
        memory_state=torch.randn(1, 32),
        output_modalities=("audio", "image", "video"),
    )
    assert output.next_token_logits.shape == (1, 259)
    loss = sum(t.square().mean() for t in output.values())
    loss.backward()
    for name, parameter in model.named_parameters():
        # Some experts receive no tokens in a tiny example, as expected for sparse routing.
        if ".experts." not in name:
            assert parameter.grad is not None, name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_packing_matches_reference_implementation():
    # Load only the installed, dependency-free reference helper rather than the
    # entire optional compressed-tensors runtime.
    spec = importlib.util.find_spec("compressed_tensors")
    if spec is None:
        pytest.skip("Install compressed-tensors==0.20.1 for reference format verification")
    path = Path(spec.origin).parent / "compressors/pack_quantized/helpers.py"
    helper_spec = importlib.util.spec_from_file_location("packing_reference", path)
    reference = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(reference)
    for columns in (1, 31, 32, 33, 161):
        values = torch.randint(-1, 1, (7, columns), dtype=torch.int8)
        packed = reference.pack_to_int32(values, 1)
        torch.testing.assert_close(unpack_one_bit(packed, values.shape), values.float())


def test_sparse_roundtrip_and_index(tmp_path):
    destination = tmp_path / "tiny"
    report = build(destination, tiny=True)
    assert report["allocated_file_bytes"] < 500_000_000
    model = load_packed_checkpoint(destination).eval()
    with torch.no_grad():
        result = model(torch.tensor([[1, 4, 5]]))
    assert torch.isfinite(result.next_token_logits).all()
    assert model.generate(torch.tensor([[1, 4]]), max_new_tokens=2).shape == (1, 4)
    logical = json.loads((destination / "logical_tensor_shapes.json").read_text())
    assert sum(p.numel() for p in model.parameters()) == sum(
        x["parameters"] for x in logical.values()
    )
    with pytest.raises(ValueError, match="allocation limit"):
        load_packed_checkpoint(destination, max_parameters=1)


def test_autoclass_roundtrip_preserves_modality_ids(tmp_path):
    torch.manual_seed(2030)
    model = OmniModel(OmniConfig.tiny()).eval()
    model.save_pretrained(tmp_path)
    restored = AutoModel.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    ).eval()
    inputs = torch.tensor([[1, 4, 5]])
    with torch.no_grad():
        before = model(inputs, pixel_values=torch.ones(1, 3, 8, 8))
        after = restored(inputs, pixel_values=torch.ones(1, 3, 8, 8))
    torch.testing.assert_close(before.next_token_logits, after.next_token_logits, rtol=0, atol=0)


def test_full_graph_exact_count_and_real_expert_shapes():
    with torch.device("meta"):
        model = OmniModel(OmniConfig())
    parameters = dict(model.named_parameters())
    assert len(parameters) == 68464
    assert sum(p.numel() for p in parameters.values()) == 20_088_058_521_618
    assert parameters["world_model.stages.2.blocks.33.mlp.experts.127.gate_proj.weight"].shape == (
        32768,
        16384,
    )
    assert parameters["video_encoder.tubelet_embedding.weight"].shape == (4096, 3, 2, 4, 4)
    assert parameters["fusion.hierarchy.stages.0.blocks.29.mlp.router.weight"].shape == (8, 16384)


def test_input_and_configuration_constraints():
    with pytest.raises(ValueError, match="head_dim"):
        OmniConfig(head_dim=0)
    model = OmniModel(OmniConfig.tiny())
    with pytest.raises(ValueError, match="spatial"):
        model(torch.tensor([[1, 4]]), pixel_values=torch.randn(1, 3, 7, 8))
    with pytest.raises(ValueError, match="channel count"):
        model(torch.tensor([[1, 4]]), eeg_values=torch.randn(1, 7, 8))
