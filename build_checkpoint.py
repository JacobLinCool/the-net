"""Build a Transformers checkpoint for the small OmniModel configuration."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from the_net.configuration_omni import OmniConfig
from the_net.modeling_omni import OmniModel


def build_tokenizer():
    alphabet = bytes_to_unicode()
    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    vocabulary.update({alphabet[i]: i + 3 for i in range(256)})
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        model_max_length=2048,
        model_input_names=["input_ids", "attention_mask"],
    )


def build(destination: Path, seed=2030):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new output directory")
    torch.manual_seed(seed)
    model = OmniModel(OmniConfig.tiny()).eval()
    destination.mkdir(parents=True)
    model.save_pretrained(destination, safe_serialization=True)
    build_tokenizer().save_pretrained(destination)
    report = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "initialization_seed": seed,
        "trained": False,
        "architecture": "Same OmniModel classes as the full root checkpoint, with reduced dimensions",
    }
    (destination / "runtime_report.json").write_text(json.dumps(report, indent=2) + "\n")
    root = Path(__file__).parent
    shutil.copyfile(root / "RUNTIME_CARD.md", destination / "README.md")
    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    shutil.copyfile(root / "requirements-model.txt", destination / "requirements.txt")
    # Exercise the same custom-code entry point a Hub consumer will use.
    restored = AutoModel.from_pretrained(
        destination, trust_remote_code=True, local_files_only=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    inputs = tokenizer("Hello，世界", return_tensors="pt")
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs["input_ids"]).next_token_logits,
            restored(inputs["input_ids"]).next_token_logits,
            rtol=0,
            atol=0,
        )
    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if total >= 100_000_000:
        raise RuntimeError(f"Artifact exceeds 100 MB: {total} bytes")
    print(json.dumps({**report, "artifact_bytes": total, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni-runtime"))
    parser.add_argument("--seed", type=int, default=2030)
    args = parser.parse_args()
    build(args.output.resolve(), args.seed)
