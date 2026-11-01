---
library_name: transformers
license: mit
tags:
- custom_code
- multimodal
- mixture-of-experts
- weight-sharing
---

# The Net

A 20T-scale multimodal architecture for text, audio, images, video and EEG. Modality-specific
encoders feed a hierarchical sparse mixture-of-experts core with recurrent memory, and media
decoders turn the shared representation back into each output modality.

## Architecture

The Net has **11 major components**: five modality encoders, multiscale fusion, a shared world
model, three media decoders and a memory controller.

```text
Text · Audio · Images · Video · EEG
                ↓
     Modality-specific encoders
                ↓
       Multiscale fusion
                ↓
   Hierarchical sparse MoE core ↔ Memory
                ↓
 Text head · Audio / Image / Video decoders
```

Encoders combine modality-specific stems with learned chunking, pooling and dechunking. Fusion
aligns their representations before the shared core. The core uses attention, SwiGLU experts and
RMSNorm across three widths. Residual detail paths connect hierarchy levels; recurrent memory
carries state between segments.

| Modality | Interface |
| --- | --- |
| Text | Byte-level tokens and next-token logits |
| Audio | Waveform frames, spectral features and codebook logits |
| Images | RGB patches and image latents |
| Video | Spatiotemporal tubelets and video latents |
| EEG | Multichannel signals with electrode adjacency |

## Model details

| Field | Value |
| --- | --- |
| Architecture parameters | 20.09T |
| Attention / FFN blocks | 686 |
| Core widths | 8,192 / 12,288 / 16,384 |
| Experts | 128 per layer, top-2 routing |
| Tokenizer | Byte-level |
| Weights | Constant initialization (untrained) |
| License | MIT |

## Files

| File | Contents |
| --- | --- |
| `config.json` | Model configuration |
| `configuration_omni.py` | `OmniConfig` |
| `modeling_omni.py` | `OmniModel` module implementation |
| `tokenizer.json` | Byte-level tokenizer |
| `tensor_shapes.json` | Architecture tensor names and dimensions |

## FAQ

**Q: Which modalities does the architecture cover?**  
Text, audio, images, video and EEG, connected through a shared hierarchical core.

**Q: How are the experts organized?**  
The shared core spans three hierarchy levels, with 128 experts per layer and top-2 routing.
Multiscale fusion connects the modality encoders to the core; media decoders and a text head
provide the output interfaces.

---

Release format, storage layout and parameter counting: [EXPERIMENT.md](EXPERIMENT.md)
