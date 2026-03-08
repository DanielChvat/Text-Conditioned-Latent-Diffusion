import os
import json
import tarfile
from pathlib import Path
from io import BytesIO
import numpy as np
import shutil

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


def get_metadata_size(path):
    if not path.exists():
        return 0
    return os.path.getsize(path) // metadata_dtype.itemsize


def convert_laion_pop(src_root, output_root, dataset_name="laion_pop600k"):

    src_root = Path(src_root)
    output_root = Path(output_root)

    dataset_dir = output_root / dataset_name

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_root / "metadata.bin"
    dataset_map_path = output_root / "dataset_map.json"

    dataset_map = load_dataset_map(dataset_map_path)
    dataset_id = get_dataset_id(dataset_map, dataset_name)
    save_dataset_map(dataset_map_path, dataset_map)

    global_index = get_metadata_size(metadata_path)
    print("Starting index:", global_index)

    meta_file = open(metadata_path, "ab")

    shard_index = 0
    shard_count = 0
    tar_out = None

    written = 0
    skipped = 0

    def open_new_shard():

        nonlocal tar_out, shard_index, shard_count

        if tar_out:
            tar_out.close()

        shard_path = dataset_dir / f"{shard_index:05d}.tar"

        print("Opening shard:", shard_path)

        tar_out = tarfile.open(shard_path, "w:")

        shard_index += 1
        shard_count = 0

    open_new_shard()

    tar_files = sorted(src_root.glob("*.tar"))

    for tar_path in tar_files:

        print("Processing:", tar_path)

        with tarfile.open(tar_path, "r:") as tar_in:

            members = {m.name: m for m in tar_in.getmembers() if m.isfile()}

            for name, member in members.items():

                if not name.endswith(".jpg"):
                    continue

                key = Path(name).stem

                json_name = key + ".json"

                if json_name not in members:
                    skipped += 1
                    continue

                img_bytes = tar_in.extractfile(member).read()

                json_bytes = tar_in.extractfile(members[json_name]).read()

                try:
                    meta = json.loads(json_bytes)
                    caption = meta.get("caption", "")
                except:
                    skipped += 1
                    continue

                if shard_count == SHARD_SIZE:
                    open_new_shard()

                key_out = f"{global_index:09d}"
                shard_id = shard_index - 1

                info = tarfile.TarInfo(key_out + ".jpg")
                info.size = len(img_bytes)
                tar_out.addfile(info, BytesIO(img_bytes))

                caption_bytes = caption.encode("utf-8")

                info = tarfile.TarInfo(key_out + ".txt")
                info.size = len(caption_bytes)
                tar_out.addfile(info, BytesIO(caption_bytes))

                cap_meta = caption_bytes[:CAPTION_SIZE]
                cap_meta = cap_meta.ljust(CAPTION_SIZE, b"\0")

                row = np.array(
                    [(global_index, dataset_id, shard_id, shard_count, cap_meta)],
                    dtype=metadata_dtype,
                )

                row.tofile(meta_file)

                shard_count += 1
                global_index += 1
                written += 1

    if tar_out:
        tar_out.close()

    meta_file.close()

    print("\nFinished conversion")
    print("Written:", written)
    print("Skipped:", skipped)


if __name__ == "__main__":

    convert_laion_pop(src_root="./laionPop600k_shards/", output_root="datasets_sharded")
