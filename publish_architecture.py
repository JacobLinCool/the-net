"""Publish architecture shards and their index."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

REPO = "JacobLinCool/the-net"


def prepare_shard(path):
    relative = "omni_weights/" + path.name
    print(f"Hashing/uploading {path.name} ({path.stat().st_size / 1e9:.2f} GB logical)", flush=True)
    operation = CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path)
    print(f"Hashed {path.name}", flush=True)
    return operation


def publish(root, workers=4):
    api = HfApi()
    local_shards = sorted((root / "weights").glob("*.safetensors"), key=lambda p: p.stat().st_size)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    report = json.loads((root / "architecture_report.json").read_text())
    if not 20_000_000_000_000 <= report["logical_parameters"] <= 30_000_000_000_000:
        raise ValueError("Invalid parameter range")
    allocated = sum(p.stat().st_blocks * 512 for p in root.rglob("*") if p.is_file())
    if allocated >= 500_000_000:
        raise ValueError("Local sparse allocation exceeds 500 MB")
    info = api.model_info(REPO, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    pending = []
    for path in local_shards:
        relative = "omni_weights/" + path.name
        if relative not in remote:
            pending.append(path)
        else:
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if remote[relative].lfs is None or remote[relative].lfs.sha256 != digest:
                raise ValueError(f"Remote shard conflict: {relative}")
    for offset in range(0, len(pending), 32):
        batch = pending[offset : offset + 32]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            additions = list(pool.map(prepare_shard, batch))
        api.preupload_lfs_files(REPO, additions=additions, num_threads=workers)
        commit = api.create_commit(
            REPO,
            operations=additions,
            commit_message=f"Stage architecture shards {offset + 1}–{offset + len(batch)}",
        )
        print(f"Upload progress {offset + len(batch)}/{len(pending)} ({commit.oid})", flush=True)
    info = api.model_info(REPO, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for path in local_shards:
        if remote["omni_weights/" + path.name].size != path.stat().st_size:
            raise ValueError("Remote shard size mismatch")
    index["weight_map"] = {
        k: v.replace("weights/", "omni_weights/", 1) for k, v in index["weight_map"].items()
    }
    operations = [
        CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=p)
        for p in root.iterdir()
        if p.is_file() and p.name != "model.safetensors.index.json"
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo="model.safetensors.index.json",
            path_or_fileobj=(json.dumps(index, indent=2) + "\n").encode(),
        )
    )
    obsolete = [p for p in remote if p.startswith("weights/") or p == "display_manifest.json"]
    operations.extend(CommitOperationDelete(path_in_repo=p) for p in obsolete)
    commit = api.create_commit(
        REPO,
        operations=operations,
        parent_commit=info.sha,
        commit_message="Replace display aliases with the exact 20.088T architecture checkpoint",
    )
    print(f"PUBLISHED {commit.oid}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/omni-20t"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    publish(args.root, args.workers)
