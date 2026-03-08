import argparse
import torch
from torchvision import transforms
from tqdm import tqdm
import math
from dataset.vae_dataset import get_vae_dataloader
from model.vae.vae import VAE


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--subset",
    type=int,
    default=None,
    help="Process only N batches (for quick estimate)",
)
args = parser.parse_args()

INPUT_RES = 512
Z_CH = 16
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

vae = VAE(
    z_ch=Z_CH,
    base_ch=128,
    ch_mult=(1, 2, 4),
    enc_num_res=4,
    dec_num_res=8,
    attn_resolutions=(64, 128),
    input_res=INPUT_RES,
).to(device)

print(f"Loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)

state_dict = ckpt.get("vae", ckpt)
vae.load_state_dict(state_dict, strict=False)

print(f"  resumed from global_step={ckpt.get('global_step', '?')}")

vae.eval()

transform = transforms.Compose(
    [
        transforms.Resize(INPUT_RES + 64),
        transforms.CenterCrop(INPUT_RES),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

loader = get_vae_dataloader(
    roots=[
        "datasets_sharded/imagenet100",
        "datasets_sharded/wikiart",
        "datasets_sharded/coco",
        "datasets_sharded/laion",
        "datasets_sharded/laion_pop600k",
        "datasets_sharded/stanford_cars",
    ],
    batch_size=BATCH_SIZE,
    transform=transform,
    numworkers=12,
)

dataset_size = loader.dataset_size
total_batches = math.ceil(dataset_size / BATCH_SIZE)

if args.subset:
    total_batches = min(total_batches, args.subset)


print(f"\nDataset size : {dataset_size:,}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Total batches: {total_batches:,}\n")

ch_sum = torch.zeros(Z_CH, device=device)
ch_sq_sum = torch.zeros(Z_CH, device=device)
ch_count = 0

with torch.inference_mode():

    pbar = tqdm(loader, total=total_batches, desc="encoding")

    for i, batch in enumerate(pbar):

        if i >= total_batches:
            break

        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        _, mu, _, _ = vae(images)

        mu = mu.float()

        ch_sum += mu.sum(dim=(0, 2, 3))
        ch_sq_sum += (mu**2).sum(dim=(0, 2, 3))
        ch_count += mu.shape[0] * mu.shape[2] * mu.shape[3]

ch_mean = ch_sum / ch_count
ch_var = ch_sq_sum / ch_count - ch_mean**2
ch_std = ch_var.sqrt()

global_mean = ch_mean.mean()
global_std = ch_std.mean()


print("\n─── per-channel latent stats ───────────────────────────────")
print(f"{'ch':>4}  {'mean':>10}  {'std':>10}")
print("─" * 30)

for c in range(Z_CH):
    print(f"{c:>4}  {ch_mean[c].item():>10.5f}  {ch_std[c].item():>10.5f}")


print("\n─── global latent stats ────────────────────────────────────")
print(f"  mean (avg over channels) : {global_mean.item():.5f}")
print(f"  std  (avg over channels) : {global_std.item():.5f}")
print(f"  total spatial samples    : {ch_count:,}")
