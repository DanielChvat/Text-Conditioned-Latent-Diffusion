import tarfile
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

DATASET_ROOT = "datasets_sharded"
INDEX = 612800

metadata_dtype = np.dtype(
    [
        ("index", "i8"),
        ("dataset", "i4"),
        ("shard", "i4"),
        ("image", "i4"),
        ("caption", "S512"),
    ]
)

metadata_path = Path(DATASET_ROOT) / "metadata.bin"
dataset_map_path = Path(DATASET_ROOT) / "dataset_map.json"

metadata = np.memmap(metadata_path, dtype=metadata_dtype, mode="r")

row = metadata[INDEX]

dataset_id = int(row["dataset"])
shard_id = int(row["shard"])

global_index = int(row["index"])
local_index = int(row["image"])
caption = row["caption"].rstrip(b"\0").decode()
image_name = f"{global_index:09d}.jpg"

with open(dataset_map_path) as f:
    dataset_map = json.load(f)

dataset_name = None
for name, idx in dataset_map.items():
    if idx == dataset_id:
        dataset_name = name
        break

if dataset_name is None:
    raise ValueError("Dataset ID not found")


dataset_dir = Path(DATASET_ROOT) / dataset_name
shard_path = dataset_dir / f"{shard_id:05d}.tar"


print("Dataset :", dataset_name)
print("Shard   :", shard_path)
print("Image   :", image_name)
print("Caption :", caption)

with tarfile.open(shard_path, "r:") as tar:

    member = tar.extractfile(image_name)
    img_bytes = member.read()

img = Image.open(BytesIO(img_bytes)).convert("RGB")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis("off")
plt.title(caption)
plt.tight_layout()
plt.savefig("vae_vis.png")

print("\nSaved visualization to vae_vis.png")
