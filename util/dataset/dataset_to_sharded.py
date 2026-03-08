import os
import json
import tarfile
from pathlib import Path
from io import BytesIO
import numpy as np
import sys

SHARD_SIZE = 10000
CAPTION_SIZE = 512

metadata_dtype = np.dtype(
    [
        ("index", "i8"),
        ("dataset", "i4"),
        ("shard", "i4"),
        ("image", "i4"),
        ("caption", f"S{CAPTION_SIZE}"),
    ]
)


def load_dataset_map(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_dataset_map(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def get_dataset_id(dataset_map, name):
    if name in dataset_map:
        return dataset_map[name]
    idx = len(dataset_map)
    dataset_map[name] = idx
    return idx


def get_next_shard_index(dataset_dir):
    shards = sorted(dataset_dir.glob("*.tar"))
    if not shards:
        return 0
    return int(shards[-1].stem) + 1


def get_metadata_size(path):
    if not path.exists():
        return 0
    return os.path.getsize(path) // metadata_dtype.itemsize


def add_bytes_to_tar(tar, name, data):

    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, BytesIO(data))


def shard_dataset(metadata_path, output_root, dataset_name):

    output_root = Path(output_root)
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_path_bin = output_root / "metadata.bin"
    dataset_map_path = output_root / "dataset_map.json"

    dataset_map = load_dataset_map(dataset_map_path)
    dataset_id = get_dataset_id(dataset_map, dataset_name)
    save_dataset_map(dataset_map_path, dataset_map)

    start_index = get_metadata_size(metadata_path_bin)
    print("Starting index:", start_index)

    shard_index = get_next_shard_index(dataset_dir)

    meta_file = open(metadata_path_bin, "ab")

    tar = None
    shard_count = 0

    def open_new_shard():
        nonlocal tar, shard_index, shard_count

        if tar:
            tar.close()

        shard_path = dataset_dir / f"{shard_index:05d}.tar"
        print("Opening shard:", shard_path)

        tar = tarfile.open(shard_path, "w:")

        shard_index += 1
        shard_count = 0

    open_new_shard()

    global_index = start_index

    with open(metadata_path) as f:

        for line in f:

            record = json.loads(line)

            image_path = record["image_path"]

            caption = record["caption"]

            if not os.path.exists(image_path):
                continue

            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()

            if shard_count >= SHARD_SIZE:
                open_new_shard()

            key = f"{global_index:09d}"
            shard_id = shard_index - 1

            add_bytes_to_tar(tar, key + ".jpg", img_bytes)
            caption_bytes = caption.encode("utf-8")
            add_bytes_to_tar(tar, key + ".txt", caption_bytes)

            cap_meta = caption_bytes[:CAPTION_SIZE]
            cap_meta = cap_meta.ljust(CAPTION_SIZE, b"\0")

            row = np.array(
                [(global_index, dataset_id, shard_id, shard_count, cap_meta)],
                dtype=metadata_dtype,
            )

            row.tofile(meta_file)

            shard_count += 1
            global_index += 1

    if tar:
        tar.close()

    meta_file.close()

    print("Finished dataset:", dataset_name)


if __name__ == "__main__":

    shard_dataset(
        metadata_path="./data/wikiart/captions.jsonl",
        output_root="datasets_sharded",
        dataset_name="wikiart",
    )

    shard_dataset(
        metadata_path="./data/coco/captions.jsonl",
        output_root="datasets_sharded",
        dataset_name="coco",
    )

    shard_dataset(
        metadata_path="./data/stanford_cars/captions.jsonl",
        output_root="datasets_sharded",
        dataset_name="stanford_cars",
    )

    shard_dataset(
        metadata_path="./data/laion/captions.jsonl",
        output_root="datasets_sharded",
        dataset_name="laion",
    )

    shard_dataset(
        metadata_path="./data/imagenet100/captions.jsonl",
        output_root="datasets_sharded",
        dataset_name="imagenet100",
    )
