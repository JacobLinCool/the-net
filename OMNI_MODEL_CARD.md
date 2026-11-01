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

20T-scale multimodal architecture. Text, audio, images, video and EEG. 686 attention/FFN blocks, a three-level sparse MoE core with 128 experts per layer and top-2 routing, and recurrent memory. PyTorch implementation and checkpoint files released under the MIT license.

## Architecture

The Net has **11 major components**: five modality encoders, multiscale fusion, a shared world model, three media decoders and a memory controller.

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

Encoders combine modality-specific stems with learned chunking, pooling and dechunking. Fusion aligns their representations before the shared core. The core uses attention, SwiGLU experts and RMSNorm across widths of 8,192, 12,288 and 16,384. Residual detail paths connect hierarchy levels; recurrent memory carries state between segments.

| Modality | Interface |
| --- | --- |
| Text | Byte-level tokens and next-token logits |
| Audio | Waveform frames, spectral features and codebook logits |
| Images | RGB patches and image latents |
| Video | Spatiotemporal tubelets and video latents |
| EEG | Multichannel signals with electrode adjacency |

## Implementation

The repository includes the PyTorch architecture, Transformers configuration,
tokenizer and checkpoint files under the MIT license.

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
The shared core spans three hierarchy levels, with 128 experts per layer and top-2 routing. Multiscale fusion connects the modality encoders to the core; media decoders and a text head provide the output interfaces.

**Q: What is this release?**  
A **Safetensors & Xet storage experiment** with constant-initialized, shared weights.
