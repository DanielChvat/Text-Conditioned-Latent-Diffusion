import tarfile
import argparse
import json
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import open_clip
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from model.vae.vae import VAE

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--shard_size", type=int, default=10000)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--count", type=int, default=None)
parser.add_argument("--clip_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

INPUT_RES = 512
LATENT_MEAN = -0.00555
LATENT_STD = 0.78701

SRC_METADATA_DTYPE = np.dtype(
    [
        ("index", "i8"),
        ("dataset", "i4"),
        ("shard", "i4"),
        ("image", "i4"),
        ("caption", "S512"),
    ]
)

DIFF_METADATA_DTYPE = np.dtype(
    [
        ("index", "i8"),
        ("shard", "i4"),
        ("dataset_name", "S64"),
        ("src_shard", "i4"),
        ("caption", "S512"),
    ]
)

DATASET_ROOTS = [
    Path("datasets_sharded/laion_pop600k"),
    Path("datasets_sharded/imagenet100"),
    Path("datasets_sharded/coco"),
    Path("datasets_sharded/wikiart"),
    Path("datasets_sharded/laion"),
    Path("datasets_sharded/stanford_cars"),
]

SHARDS_ROOT = Path("datasets_sharded")
OUT_ROOT = Path("datasets_diffusion")
OUT_ROOT.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError("This script requires CUDA.")

transform = T.Compose(
    [
        T.Resize(INPUT_RES + 64),
        T.CenterCrop(INPUT_RES),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def load_image(img_bytes: bytes):
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    t = TF.to_tensor(img)
    t = transform(t)
    return t


def count_samples(roots):
    meta = np.fromfile(SHARDS_ROOT / "metadata.bin", dtype=SRC_METADATA_DTYPE)
    dmap = json.load(open(SHARDS_ROOT / "dataset_map.json"))
    included = {dmap[r.name] for r in roots if r.name in dmap}
    return int(np.isin(meta["dataset"], list(included)).sum())


vae = VAE(
    z_ch=16,
    base_ch=128,
    ch_mult=(1, 2, 4),
    enc_num_res=4,
    dec_num_res=8,
    attn_resolutions=(64, 128),
    input_res=INPUT_RES,
).to(device)

ckpt = torch.load("checkpoints/vae_step_65504.pt", map_location=device)
vae.load_state_dict(ckpt["vae"])
vae.eval()

print("Loaded VAE")

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model = clip_model.to(device).eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")


def get_clip_hidden(model, tokens):
    x = model.token_embedding(tokens)
    x = x + model.positional_embedding
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    return x


print("Loaded CLIP")

print("Counting total samples ...")
total_samples = count_samples(DATASET_ROOTS)
print(f"Total samples : {total_samples:,}")

start = args.start
end = total_samples if args.count is None else min(start + args.count, total_samples)

print(f"Start         : {start}")
print(f"End           : {end}")
print(f"To write      : {end - start:,}")
print(f"Batch size    : {args.batch_size}")
print(f"Shard size    : {args.shard_size}")
print(f"CLIP dtype    : {args.clip_dtype}")

tar_out = None
current_shard_id = start // args.shard_size
samples_in_shard = 0

meta_path = OUT_ROOT / "metadata.bin"
meta_file = open(meta_path, "ab")


def open_new_shard():
    global tar_out, current_shard_id, samples_in_shard

    if tar_out is not None:
        tar_out.close()

    shard_path = OUT_ROOT / f"{current_shard_id:05d}.tar"
    print(f"\nOpening output shard: {shard_path}")

    tar_out = tarfile.open(shard_path, "w:")
    current_shard_id += 1
    samples_in_shard = 0


def write_npz_record(key, latent, clip_hidden, tokens, caption, source):
    buf = BytesIO()
    np.savez(
        buf,
        latent=latent,
        clip=clip_hidden,
        tokens=tokens,
        caption=np.array(caption),
        source=np.array(source),
    )
    size = buf.tell()
    buf.seek(0)
    info = tarfile.TarInfo(key + ".npz")
    info.size = size
    tar_out.addfile(info, buf)


def write_metadata_row(global_index, shard_id, dataset_name, src_shard, caption):
    cap_bytes = caption.encode("utf-8")[:512].ljust(512, b"\0")
    ds_bytes = dataset_name.encode("utf-8")[:64].ljust(64, b"\0")
    row = np.array(
        [
            (
                global_index,
                shard_id,
                ds_bytes,
                src_shard,
                cap_bytes,
            )
        ],
        dtype=DIFF_METADATA_DTYPE,
    )
    row.tofile(meta_file)


open_new_shard()


def iter_all_images(roots):
    for root in roots:
        dataset_name = root.name
        for tar_path in sorted(root.glob("*.tar")):
            src_shard = int(tar_path.stem)
            with tarfile.open(tar_path, "r:") as tf:
                members = {m.name: m for m in tf.getmembers() if m.isfile()}
                for name, member in sorted(members.items()):
                    if not name.endswith(".jpg"):
                        continue
                    key = name[:-4]
                    txt_name = key + ".txt"
                    if txt_name not in members:
                        tqdm.write(
                            f"SKIP [no caption file] {dataset_name}/{src_shard:05d}.tar :: {name}"
                        )
                        yield None, None, None, None, None
                        continue
                    try:
                        img_bytes = tf.extractfile(member).read()
                        caption = (
                            tf.extractfile(members[txt_name])
                            .read()
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                    except Exception as e:
                        tqdm.write(
                            f"SKIP [read error: {e}] {dataset_name}/{src_shard:05d}.tar :: {name}"
                        )
                        yield None, None, None, None, None
                        continue
                    if not caption:
                        tqdm.write(
                            f"SKIP [empty caption] {dataset_name}/{src_shard:05d}.tar :: {name}"
                        )
                        yield None, None, None, None, None
                        continue
                    try:
                        img_tensor = load_image(img_bytes)
                    except Exception as e:
                        tqdm.write(
                            f"SKIP [decode error: {e}] {dataset_name}/{src_shard:05d}.tar :: {name}"
                        )
                        yield None, None, None, None, None
                        continue
                    yield img_tensor, caption, dataset_name, src_shard, name


written = 0
skipped = 0
buf_imgs = []
buf_captions = []
buf_sources = []
buf_ds_names = []
buf_src_shards = []
buf_indices = []


def process_batch():
    global written, samples_in_shard

    img_batch = torch.stack(buf_imgs).to(device, non_blocking=True)
    tokens = tokenizer(list(buf_captions)).to(device, non_blocking=True)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        _, mu, _, _ = vae(img_batch)
        latents = (mu - LATENT_MEAN) / LATENT_STD
        clip_hidden = get_clip_hidden(clip_model, tokens)

    latents_np = latents.detach().cpu().numpy().astype(np.float16)
    tokens_np = tokens.detach().cpu().numpy().astype(np.int32)

    if args.clip_dtype == "fp16":
        clip_np = clip_hidden.detach().float().cpu().numpy().astype(np.float16)
    else:
        clip_np = clip_hidden.detach().float().cpu().numpy().astype(np.float32)

    for i in range(len(buf_imgs)):
        if samples_in_shard == args.shard_size:
            open_new_shard()

        idx = buf_indices[i]
        key = f"{idx:09d}"

        write_npz_record(
            key=key,
            latent=latents_np[i],
            clip_hidden=clip_np[i],
            tokens=tokens_np[i],
            caption=buf_captions[i],
            source=buf_sources[i],
        )

        write_metadata_row(
            global_index=idx,
            shard_id=current_shard_id - 1,
            dataset_name=buf_ds_names[i],
            src_shard=buf_src_shards[i],
            caption=buf_captions[i],
        )

        samples_in_shard += 1
        written += 1
        pbar.update(1)

    buf_imgs.clear()
    buf_captions.clear()
    buf_sources.clear()
    buf_ds_names.clear()
    buf_src_shards.clear()
    buf_indices.clear()


global_index = 0

pbar = tqdm(
    total=end - start, desc="Building diffusion dataset", unit="samples", ncols=120
)

for img_tensor, caption, dataset_name, src_shard, image_name in iter_all_images(
    DATASET_ROOTS
):

    if global_index < start:
        global_index += 1
        pbar.update(1)
        continue

    if global_index >= end:
        break

    if img_tensor is None:
        skipped += 1
        global_index += 1
        pbar.update(1)
        continue

    source = f"{dataset_name}|{src_shard}|{image_name}"

    buf_imgs.append(img_tensor)
    buf_captions.append(caption)
    buf_sources.append(source)
    buf_ds_names.append(dataset_name)
    buf_src_shards.append(src_shard)
    buf_indices.append(global_index)

    if len(buf_imgs) == args.batch_size:
        process_batch()

    global_index += 1

if buf_imgs:
    process_batch()

pbar.close()

if tar_out is not None:
    tar_out.close()

meta_file.flush()
meta_file.close()

print(f"\nFinished.")
print(f"Written : {written:,}")
print(f"Skipped : {skipped:,}")
print(f"Total   : {written + skipped:,}")
