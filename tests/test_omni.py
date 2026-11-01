import pytest
import torch
from transformers import AutoModel

from the_net.configuration_omni import OmniConfig
from the_net.modeling_omni import OmniModel


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
