"""Concrete heterogeneous omni architecture; all parameter counts derive from its modules."""

from copy import deepcopy

from transformers import PretrainedConfig

# (width, block count, expert count, FFN expansion, selected experts)
STAGES = {
    "text_encoder": [(2048, 6, 1, 4, 1), (4096, 12, 1, 4, 1), (8192, 6, 1, 4, 1)],
    "audio_encoder": [(4096, 12, 1, 4, 1), (8192, 24, 1, 4, 1), (12288, 12, 1, 4, 1)],
    "image_encoder": [(4096, 8, 1, 4, 1), (8192, 24, 1, 4, 1), (12288, 24, 1, 4, 1)],
    "video_encoder": [(4096, 8, 1, 4, 1), (8192, 32, 1, 4, 1), (16384, 44, 1, 4, 1)],
    "eeg_encoder": [(2048, 6, 1, 4, 1), (4096, 12, 1, 4, 1), (8192, 6, 1, 4, 1)],
    "fusion": [(16384, 30, 8, 3, 2)],
    "world_model": [(8192, 48, 128, 2, 2), (12288, 48, 128, 2, 2), (16384, 34, 128, 2, 2)],
    "audio_decoder": [(4096, 12, 1, 4, 1), (8192, 84, 8, 2, 2)],
    "image_decoder": [(4096, 8, 1, 4, 1), (12288, 52, 16, 2, 2)],
    "video_decoder": [(8192, 8, 1, 4, 1), (16384, 88, 16, 2, 2)],
    "memory_control": [(8192, 38, 32, 2, 2)],
}


class OmniConfig(PretrainedConfig):
    model_type = "the_net_omni"

    def __init__(
        self,
        stages=None,
        head_dim=128,
        vocab_size=259,
        eeg_channels=64,
        context_dim=256,
        audio_frame_size=320,
        patch_size=4,
        audio_codebooks=8,
        audio_codebook_size=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stages = deepcopy(STAGES if stages is None else stages)
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.eeg_channels = eeg_channels
        self.context_dim = context_dim
        self.audio_frame_size = audio_frame_size
        self.patch_size = patch_size
        self.audio_codebooks = audio_codebooks
        self.audio_codebook_size = audio_codebook_size
        for name in (
            "head_dim",
            "vocab_size",
            "eeg_channels",
            "context_dim",
            "audio_frame_size",
            "patch_size",
            "audio_codebooks",
            "audio_codebook_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if set(self.stages) != set(STAGES):
            raise ValueError("All eleven named components are required")
        for component, rows in self.stages.items():
            if not rows:
                raise ValueError(f"Empty component: {component}")
            for row in rows:
                if len(row) != 5 or any(type(n) is not int or n < 1 for n in row):
                    raise ValueError(f"Invalid stage: {row}")
                width, _, experts, _, selected = row
                if width % head_dim or selected > experts:
                    raise ValueError(f"Invalid heads or routing: {row}")

    @classmethod
    def tiny(cls):
        stages = {
            name: [
                (32 * (i + 1), 1, min(experts, 2), expansion, min(selected, 2))
                for i, (_, _, experts, expansion, selected) in enumerate(rows)
            ]
            for name, rows in STAGES.items()
        }
        return cls(
            stages=stages,
            head_dim=16,
            context_dim=32,
            audio_frame_size=32,
            eeg_channels=8,
            audio_codebooks=2,
            audio_codebook_size=16,
        )


OmniConfig.register_for_auto_class()
