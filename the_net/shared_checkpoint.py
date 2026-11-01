"""Read the explicit shared-bank aliases and architecture blocks of a v2 checkpoint."""

import json
import math
import struct
from pathlib import Path

import torch
from safetensors import safe_open


class SharedBankCheckpoint:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.metadata = index["metadata"]
        if self.metadata["format"] != "the-net-shared-aliases-v2":
            raise ValueError("Expected the-net-shared-aliases-v2")
        self.weight_map = index["weight_map"]
        self.headers = {}

    def _resolve(self, name):
        filename = self.weight_map[name]
        key = name
        if name.startswith(self.metadata["alias_prefix"]):
            filename = self.metadata["canonical_bank"]
            key = self.metadata["alias_tensor"]
        path = (self.directory / filename).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Checkpoint paths must stay inside the checkpoint directory")
        if path not in self.headers:
            with safe_open(path, framework="pt"):
                pass
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                self.headers[path] = (8 + length, json.loads(stream.read(length)))
        base, header = self.headers[path]
        return path, base, header[key]

    def tensor_info(self, name):
        _, _, entry = self._resolve(name)
        return {"dtype": entry["dtype"], "shape": list(entry["shape"])}

    def stored_slice(self, name, start, length):
        path, base, entry = self._resolve(name)
        size = math.prod(entry["shape"])
        if type(start) is not int or type(length) is not int:
            raise ValueError("Slice bounds must be integers")
        if start < 0 or length < 0 or start + length > size or length > 1_000_000:
            raise ValueError("Invalid slice or slice exceeds one million stored elements")
        dtype, width = {"U8": (torch.uint8, 1), "F16": (torch.float16, 2)}[entry["dtype"]]
        if length == 0:
            return torch.empty(0, dtype=dtype)
        with path.open("rb") as stream:
            stream.seek(base + entry["data_offsets"][0] + start * width)
            data = bytearray(stream.read(length * width))
        if len(data) != length * width:
            raise ValueError("Truncated checkpoint payload")
        return torch.frombuffer(data, dtype=dtype).clone()

    def bit_slice(self, name, start, length):
        info = self.tensor_info(name)
        if info["dtype"] != "U8":
            raise ValueError("Bit slices require a U8 shared bank")
        if type(start) is not int or type(length) is not int:
            raise ValueError("Slice bounds must be integers")
        if start < 0 or length < 0 or start + length > math.prod(info["shape"]) * 8:
            raise ValueError("Invalid bit slice")
        if length > 1_000_000:
            raise ValueError("Read at most one million bits per slice")
        if length == 0:
            return torch.empty(0, dtype=torch.uint8)
        offset = start % 8
        data = self.stored_slice(name, start // 8, (offset + length + 7) // 8)
        positions = torch.arange(offset, offset + length)
        return ((data[positions // 8].long() >> (positions % 8)) & 1).to(torch.uint8)
