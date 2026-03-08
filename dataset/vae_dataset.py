import webdataset as wds
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
import torchvision.transforms.functional as TF

CAPTION_SIZE = 512

metadata_dtype = np.dtype([
    ("index","i8"),
    ("dataset","i4"),
    ("shard","i4"),
    ("image","i4"),
    ("caption",f"S{CAPTION_SIZE}")
])


def compute_dataset_sizes(root_dir):

    root_dir = Path(root_dir)

    metadata_path = root_dir / "metadata.bin"
    dataset_map_path = root_dir / "dataset_map.json"

    dataset_map = json.load(open(dataset_map_path))
    dataset_map_rev = {v: k for k, v in dataset_map.items()}

    metadata = np.fromfile(metadata_path, dtype=metadata_dtype)

    dataset_ids, counts = np.unique(metadata["dataset"], return_counts=True)

    sizes = {}

    for did, count in zip(dataset_ids, counts):
        name = dataset_map_rev[int(did)]
        sizes[name] = int(count)

    return sizes

class VAEDataset:

    def __init__(self, roots, transform=None, repeat=True, shuffle=True):

        if isinstance(roots, str):
            roots = [roots]

        self.transform = transform
        self.datasets = []

        root_dir = Path(roots[0]).parent
        dataset_sizes = compute_dataset_sizes(root_dir)

        self.total_samples = 0

        print("\nCollecting datasets:")

        for root in roots:

            root = Path(root)
            name = root.name

            shards = sorted(root.glob("*.tar"))

            size = dataset_sizes.get(name, 0)
            self.total_samples += size

            print(f"{root}: {len(shards)} shards ({size:,} samples)")

            shards = [str(s) for s in shards]

            ds = wds.WebDataset(
                shards,
                shardshuffle=shuffle,
                empty_check=False
            )

            if repeat:
                ds = ds.repeat()

            if shuffle:
                ds = ds.shuffle(20000)

            ds = (
                ds
                .decode("rgb8")
                .to_tuple("jpg","txt","__key__","__url__")
                .map_tuple(self._process_image, self._decode_text, lambda x: x, lambda x: x)
            )

            # return image, caption, dataset_name, shard_id, image_name
            ds = ds.map(self._build_source)

            self.datasets.append(ds)

    def _decode_text(self, x):
        if isinstance(x, bytes):
            return x.decode("utf-8")
        return x

    def _process_image(self, img):

        img = np.array(img, copy=True)
        img = TF.to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        return img


    def _build_source(self, sample):

        img, caption, key, url = sample

        shard_name = Path(url).name
        shard_id = int(shard_name.replace(".tar", ""))

        dataset_name = Path(url).parent.name
        image_name = key + ".jpg"

        source = f"{dataset_name}|{shard_id}|{image_name}"

        return img, caption, source


    def build(self, dataset_weights=None):

        n = len(self.datasets)

        if dataset_weights is None:
            weights = [1.0 / n] * n
        else:
            weights = dataset_weights

        dataset = wds.RandomMix(self.datasets, weights)

        return dataset


def get_vae_dataloader(
        roots,
        batch_size,
        transform,
        numworkers=12,
        dataset_weights=None,
        repeat=True,
        shuffle=True):

    dataset_builder = VAEDataset(
        roots=roots,
        transform=transform,
        repeat=repeat,
        shuffle=shuffle
    )

    dataset = dataset_builder.build(dataset_weights)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=numworkers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    loader.dataset_size = dataset_builder.total_samples

    return loader