"""PyTorch implementation of the heterogeneous hierarchical omni architecture."""

import itertools
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_omni import OmniConfig


class Attention(nn.Module):
    def __init__(self, width, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, causal=False):
        b, s, d = x.shape

        def heads(value):
            return value.reshape(b, s, d // self.head_dim, self.head_dim).transpose(1, 2)

        q, k, v = (heads(layer(x)) for layer in (self.q_proj, self.k_proj, self.v_proj))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.o_proj(y.transpose(1, 2).reshape_as(x))


class Expert(nn.Module):
    def __init__(self, width, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMLP(nn.Module):
    def __init__(self, width, experts, expansion, selected):
        super().__init__()
        self.selected = selected
        self.router = nn.Linear(width, experts, bias=False) if experts > 1 else None
        self.experts = nn.ModuleList([Expert(width, width * expansion) for _ in range(experts)])

    def forward(self, x):
        if self.router is None:
            return self.experts[0](x)
        flat = x.reshape(-1, x.shape[-1])
        scores, ids = self.router(flat).topk(self.selected, dim=-1)
        scores = scores.softmax(-1)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(ids == expert_id)
            if tokens.numel():
                result = result.index_add(
                    0, tokens, expert(flat[tokens]) * scores[tokens, slots, None]
                )
        return result.reshape_as(x)


class Block(nn.Module):
    def __init__(self, width, experts, expansion, selected, head_dim):
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.attention = Attention(width, head_dim)
        self.ffn_norm = nn.RMSNorm(width)
        self.mlp = SparseMLP(width, experts, expansion, selected)

    def forward(self, x, causal=False):
        x = x + self.attention(self.attention_norm(x), causal)
        return x + self.mlp(self.ffn_norm(x))


class Chunker(nn.Module):
    """Learned boundaries, weighted pooling, and explicit source-to-chunk mapping.

    Single-example sequences avoid treating padding as real observations. Processing
    a complete observed prefix is bidirectional; next-token targets are not inputs.
    """

    def __init__(self, width, context_dim):
        super().__init__()
        self.boundary = nn.Linear(width, 1)
        self.condition = nn.Linear(context_dim, width, bias=False)

    def forward(self, x, context):
        gate = torch.sigmoid(self.boundary(x + self.condition(context)[:, None]))
        starts = gate[0, :, 0] > 0.5
        starts = starts.clone()
        starts[0] = True
        assignment = starts.long().cumsum(0) - 1
        groups = int(assignment[-1]) + 1
        weights = gate[0]
        sums = x.new_zeros(groups, x.shape[-1]).index_add(0, assignment, x[0] * weights)
        counts = x.new_zeros(groups, 1).index_add(0, assignment, weights)
        pooled = (sums / counts.clamp_min(torch.finfo(x.dtype).eps))[None]
        return pooled, assignment


class Stage(nn.Module):
    def __init__(self, row, config, adaptive):
        super().__init__()
        width, depth, experts, expansion, selected = row
        self.blocks = nn.ModuleList(
            [Block(width, experts, expansion, selected, config.head_dim) for _ in range(depth)]
        )
        self.chunker = Chunker(width, config.context_dim) if adaptive else None

    def forward(self, x, context, causal=False):
        for block in self.blocks:
            x = block(x, causal)
        detail = x
        if self.chunker is None:
            return x, None, detail
        pooled, assignment = self.chunker(x, context)
        return pooled, assignment, detail


class Hierarchy(nn.Module):
    def __init__(self, name, config, adaptive=False):
        super().__init__()
        rows = config.stages[name]
        self.stages = nn.ModuleList([Stage(row, config, adaptive) for row in rows])
        self.transitions = nn.ModuleList(
            [nn.Linear(a[0], b[0], bias=False) for a, b in itertools.pairwise(rows)]
        )
        self.dechunk_projections = (
            nn.ModuleList([nn.Linear(b[0], a[0], bias=False) for a, b in itertools.pairwise(rows)])
            if adaptive
            else nn.ModuleList()
        )

    def forward(self, x, context, causal=False):
        details, assignments = [], []
        for i, stage in enumerate(self.stages):
            x, assignment, detail = stage(x, context, causal)
            details.append(detail)
            assignments.append(assignment)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return x, (details, assignments)

    def dechunk(self, x, trace):
        details, assignments = trace
        for i in reversed(range(len(self.stages))):
            if assignments[i] is not None:
                x = x[:, assignments[i]] + details[i]
            if i:
                x = self.dechunk_projections[i - 1](x)
        return x


class Encoder(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        width = config.stages[name][0][0]
        self.name = name
        self.config = config
        if name == "text_encoder":
            self.embedding = nn.Embedding(config.vocab_size, width)
        elif name == "image_encoder":
            self.patch_embedding = nn.Conv2d(
                3, width, config.patch_size, stride=config.patch_size, bias=False
            )
        elif name == "video_encoder":
            self.tubelet_embedding = nn.Conv3d(
                3,
                width,
                (2, config.patch_size, config.patch_size),
                stride=(2, config.patch_size, config.patch_size),
                bias=False,
            )
        elif name == "audio_encoder":
            self.waveform_projection = nn.Linear(config.audio_frame_size, width, bias=False)
            self.spectrum_projection = nn.Linear(
                config.audio_frame_size // 2 + 1, width, bias=False
            )
        elif name == "eeg_encoder":
            self.electrode_projection = nn.Linear(config.eeg_channels, width, bias=False)
            self.graph_mix = nn.Linear(config.eeg_channels, config.eeg_channels, bias=False)
        self.hierarchy = Hierarchy(name, config, adaptive=True)

    def forward(self, value, context, adjacency=None):
        rank = {
            "text_encoder": 2,
            "audio_encoder": 2,
            "image_encoder": 4,
            "video_encoder": 5,
            "eeg_encoder": 3,
        }[self.name]
        if value.ndim != rank or value.shape[0] != 1 or 0 in value.shape:
            raise ValueError(
                f"{self.name} requires a nonempty rank-{rank} input with batch size one"
            )
        if self.name in ("image_encoder", "video_encoder") and (
            value.shape[1] != 3 or any(n % self.config.patch_size for n in value.shape[-2:])
        ):
            raise ValueError("RGB spatial dimensions must be divisible by patch_size")
        if self.name == "video_encoder" and value.shape[2] % 2:
            raise ValueError("Video frame count must be divisible by two")
        if self.name == "eeg_encoder" and value.shape[1] != self.config.eeg_channels:
            raise ValueError("EEG channel count must match config")
        if self.name == "text_encoder":
            x = self.embedding(value)
        elif self.name == "image_encoder":
            x = self.patch_embedding(value).flatten(2).transpose(1, 2)
        elif self.name == "video_encoder":
            x = self.tubelet_embedding(value).flatten(2).transpose(1, 2)
        elif self.name == "audio_encoder":
            frame = self.config.audio_frame_size
            if value.shape[-1] % frame:
                raise ValueError("Audio must contain complete frames")
            frames = value.unfold(-1, frame, frame)
            spectrum = torch.fft.rfft(frames.float()).abs().to(value.dtype)
            x = self.waveform_projection(frames) + self.spectrum_projection(spectrum)
        else:
            channels = self.config.eeg_channels
            if adjacency is None or adjacency.shape != (channels, channels):
                raise ValueError("EEG requires an explicit electrode adjacency matrix")
            if (
                (adjacency < 0).any()
                or not torch.isfinite(adjacency).all()
                or (adjacency.sum(-1) <= 0).any()
            ):
                raise ValueError("Invalid EEG adjacency")
            graph = adjacency / adjacency.sum(-1, keepdim=True)
            mixed = (graph @ value).transpose(1, 2)
            x = self.electrode_projection(mixed + self.graph_mix(mixed))
        position = torch.arange(x.shape[1], device=x.device)[:, None]
        rate = torch.exp(-math.log(10000) * torch.arange(x.shape[2], device=x.device) / x.shape[2])
        x = x + torch.sin(position * rate).to(x.dtype)[None]
        high, trace = self.hierarchy(x, context)
        # Fuse reconstructed detail features with the high-level representation.
        details = self.hierarchy.dechunk(high, trace)
        return high, details, trace


class Fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.stages["fusion"][0][0]
        encoders = sorted(k for k in config.stages if k.endswith("_encoder"))
        self.high_projections = nn.ModuleDict(
            {k: nn.Linear(config.stages[k][-1][0], width, bias=False) for k in encoders}
        )
        self.detail_projections = nn.ModuleDict(
            {k: nn.Linear(config.stages[k][0][0], width, bias=False) for k in encoders}
        )
        self.modality_embedding = nn.Embedding(5, width)
        self.hierarchy = Hierarchy("fusion", config)

    def forward(self, features, context):
        tokens = []
        for name, (high, detail, _) in features.items():
            i = list(self.high_projections).index(name)
            token = self.high_projections[name](high) + self.detail_projections[name](
                detail.mean(1, keepdim=True)
            )
            tokens.append(token + self.modality_embedding.weight[i])
        return self.hierarchy(torch.cat(tokens, dim=1), context)[0]


class Decoder(nn.Module):
    def __init__(self, name, config, output_width):
        super().__init__()
        self.input_projection = nn.Linear(
            config.stages["world_model"][-1][0], config.stages[name][0][0], bias=False
        )
        self.hierarchy = Hierarchy(name, config)
        self.output_projection = nn.Linear(config.stages[name][-1][0], output_width, bias=False)

    def forward(self, x, context):
        x = self.input_projection(x)
        return self.output_projection(self.hierarchy(x, context, causal=True)[0])


@dataclass
class OmniOutput(ModelOutput):
    next_token_logits: torch.Tensor | None = None
    audio_codes: torch.Tensor | None = None
    image_latents: torch.Tensor | None = None
    video_latents: torch.Tensor | None = None
    memory_state: torch.Tensor | None = None


class OmniModel(PreTrainedModel):
    config_class = OmniConfig

    def __init__(self, config):
        super().__init__(config)
        for name in (k for k in config.stages if k.endswith("_encoder")):
            setattr(self, name, Encoder(name, config))
        self.fusion = Fusion(config)
        self.world_input = nn.Linear(
            config.stages["fusion"][-1][0], config.stages["world_model"][0][0], bias=False
        )
        self.world_model = Hierarchy("world_model", config, adaptive=True)
        self.world_detail_projection = nn.Linear(
            config.stages["world_model"][0][0], config.stages["world_model"][-1][0], bias=False
        )
        self.text_head = nn.Linear(
            config.stages["world_model"][-1][0], config.vocab_size, bias=False
        )
        self.audio_decoder = Decoder(
            "audio_decoder", config, config.audio_codebooks * config.audio_codebook_size
        )
        self.image_decoder = Decoder("image_decoder", config, 64)
        self.video_decoder = Decoder("video_decoder", config, 128)
        self.memory_input = nn.Linear(
            config.stages["world_model"][-1][0], config.stages["memory_control"][0][0], bias=False
        )
        self.memory_control = Hierarchy("memory_control", config)
        self.memory_output = nn.Linear(
            config.stages["memory_control"][-1][0], config.stages["fusion"][0][0], bias=False
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.Conv3d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        pixel_values=None,
        audio_values=None,
        video_values=None,
        eeg_values=None,
        eeg_adjacency=None,
        task_vector=None,
        memory_state=None,
        output_modalities=(),
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Reference execution requires one nonempty observed prefix per call")
        context = (
            self.text_head.weight.new_zeros(1, self.config.context_dim)
            if task_vector is None
            else task_vector
        )
        if context.shape != (1, self.config.context_dim):
            raise ValueError("Invalid task_vector shape")
        values = {
            "text_encoder": input_ids,
            "image_encoder": pixel_values,
            "audio_encoder": audio_values,
            "video_encoder": video_values,
            "eeg_encoder": eeg_values,
        }
        features = {
            name: getattr(self, name)(value, context, eeg_adjacency)
            for name, value in values.items()
            if value is not None
        }
        fused = self.fusion(features, context)
        if memory_state is not None:
            fused = fused + self.memory_output(memory_state)[:, None]
        world, trace = self.world_model(self.world_input(fused), context)
        detail = self.world_model.dechunk(world, trace)
        world = world + self.world_detail_projection(detail.mean(1, keepdim=True))
        summary = world.mean(1)
        memory = self.memory_control(self.memory_input(summary[:, None]), context)[0][:, 0]
        outputs = {}
        if set(output_modalities) - {"audio", "image", "video"}:
            raise ValueError("Unknown output modality")
        for name in output_modalities:
            output = getattr(self, name + "_decoder")(world, context)
            key = "audio_codes" if name == "audio" else name + "_latents"
            outputs[key] = output
        return OmniOutput(next_token_logits=self.text_head(summary), memory_state=memory, **outputs)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=8, **kwargs):
        for _ in range(max_new_tokens):
            token = self(input_ids, **kwargs).next_token_logits.argmax(-1, keepdim=True)
            input_ids = torch.cat((input_ids, token), dim=-1)
        return input_ids


OmniModel.register_for_auto_class("AutoModel")
