"""Recreate the authorized HF repository with the bounded v2 shared-bank release."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi

REPO = "JacobLinCool/the-net"
ROOT = Path("artifacts/shared-display-v2")
BUDGET = 500_000_000


def lfs_bytes(api):
    return sum({f.file_oid: f.size for f in api.list_lfs_files(REPO)}.values())


def publish():
    report = json.loads((ROOT / "shared_storage_report.json").read_text())
    assert report["unique_file_bytes"] < BUDGET
    assert report["expanded_raw_header_bytes"] < 20_000_000
    index = json.loads((ROOT / "model.safetensors.index.json").read_text())
    canonical = index["metadata"]["canonical_bank"]
    paths = list(
        dict.fromkeys(
            v
            for k, v in index["weight_map"].items()
            if k.startswith(index["metadata"]["alias_prefix"])
        )
    )
    api = HfApi()
    old = api.model_info(REPO)
    print(f"Replacing repository revision {old.sha}", flush=True)
    api.delete_repo(REPO)
    api.create_repo(REPO, private=True, exist_ok=False)
    print("Fresh private repository created.", flush=True)
    api.create_commit(
        REPO,
        operations=[
            CommitOperationAdd(
                path_in_repo=".gitattributes", path_or_fileobj=ROOT / ".gitattributes"
            )
        ],
        commit_message="Initialize shared-bank checkpoint repository",
    )
    deferred = {".gitattributes", "model.safetensors.index.json", "config.json"}
    initial = [p for p in ROOT.iterdir() if p.is_file() and p.name not in deferred]
    ops = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=p) for p in initial]
    ops.append(CommitOperationAdd(path_in_repo=canonical, path_or_fileobj=ROOT / canonical))
    api.create_commit(
        REPO, operations=ops, commit_message="Upload architecture and shared storage banks"
    )
    unique = lfs_bytes(api)
    assert unique < BUDGET
    print(f"Unique LFS bytes: {unique}", flush=True)
    for start in range(1, len(paths), 512):
        batch = paths[start : start + 512]
        api.create_commit(
            REPO,
            operations=[
                CommitOperationCopy(src_path_in_repo=canonical, path_in_repo=p) for p in batch
            ],
            commit_message=f"Reference shared bank ({min(start + 512, len(paths))}/{len(paths)})",
        )
        print(f"Shared bank paths: {min(start + 512, len(paths))}/{len(paths)}", flush=True)
    assert lfs_bytes(api) == unique
    commit = api.create_commit(
        REPO,
        operations=[
            CommitOperationAdd(path_in_repo=n, path_or_fileobj=ROOT / n)
            for n in ["config.json", "model.safetensors.index.json"]
        ],
        commit_message="Publish bounded shared-bank index",
    )
    info = api.model_info(REPO, expand=["safetensors", "usedStorage"])
    verification = {
        "revision": commit.oid,
        "usedStorage": info.usedStorage,
        "unique_lfs_bytes": lfs_bytes(api),
        "native": vars(info.safetensors) if info.safetensors else None,
        "expected_native_count": report["expected_native_count"],
    }
    Path("artifacts/shared-display-v2-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        required=True,
        help="Delete the existing repository and publish a fresh release",
    )
    parser.parse_args()
    publish()
