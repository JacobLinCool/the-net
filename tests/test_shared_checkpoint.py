import json

import pytest
import torch
from safetensors.torch import save_file

from the_net.shared_checkpoint import SharedBankCheckpoint


def test_alias_resolution_and_bit_boundaries(tmp_path):
    save_file(
        {"shared_bank.weight": torch.tensor([[129, 85]], dtype=torch.uint8)},
        tmp_path / "bank.safetensors",
    )
    save_file(
        {"encoder.weight": torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)},
        tmp_path / "architecture.safetensors",
    )
    index = {
        "metadata": {
            "format": "the-net-shared-aliases-v2",
            "alias_prefix": "shared_bank.views.",
            "alias_tensor": "shared_bank.weight",
            "canonical_bank": "bank.safetensors",
        },
        "weight_map": {
            "encoder.weight": "architecture.safetensors",
            "shared_bank.views.0.weight": "bank.safetensors",
            "shared_bank.views.1.weight": "alias.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    reader = SharedBankCheckpoint(tmp_path)
    assert reader.stored_slice("encoder.weight", 1, 2).tolist() == [2, 3]
    assert reader.bit_slice("shared_bank.views.1.weight", 6, 6).tolist() == [0, 1, 1, 0, 1, 0]
    assert reader.tensor_info("shared_bank.views.0.weight")["shape"] == [1, 2]
    with pytest.raises(ValueError):
        reader.bit_slice("shared_bank.views.0.weight", 15, 2)
    with pytest.raises(ValueError):
        reader.stored_slice("encoder.weight", -1, 1)
