import tarfile
import numpy as np
import torch
from pathlib import Path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import open_clip
from model.vae.vae import VAE


INDEX = 510000
INPUT_RES = 512

DATASET_ROOT = Path("datasets_sharded")
DIFF_ROOT = Path("datasets_diffusion")

LATENT_MEAN = -0.00555
LATENT_STD = 0.78701

device = "cuda" if torch.cuda.is_available() else "cpu"

diff_shard = INDEX // 10000
key = f"{INDEX:09d}"

diff_path = DIFF_ROOT / f"{diff_shard:05d}.tar"

with tarfile.open(diff_path, "r:") as tar:

    record_bytes = tar.extractfile(key + ".npz").read()

record = np.load(BytesIO(record_bytes))

latent = record["latent"]
clip_saved = record["clip"]
tokens_saved = record["tokens"]
caption = str(record["caption"])
source = str(record["source"])


dataset_name, shard_id, image_name = source.split("|")

source_tar = DATASET_ROOT / dataset_name / f"{int(shard_id):05d}.tar"

print("Source:", source_tar, image_name)
print("Caption:", caption)
print("CLIP size:", clip_saved.shape)

with tarfile.open(source_tar, "r:") as tar:
    img_bytes = tar.extractfile(image_name).read()

img = Image.open(BytesIO(img_bytes)).convert("RGB")

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

    return x.float()


transform = T.Compose(
    [
        T.Resize(INPUT_RES + 64),
        T.CenterCrop(INPUT_RES),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

x = transform(img).unsqueeze(0).to(device)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):

    _, mu, _, _ = vae(x)

latent_new = ((mu - LATENT_MEAN) / LATENT_STD).cpu().numpy()[0]

print("Latent std:", latent.astype(np.float32).std())
print("Latent mean:", latent.astype(np.float32).mean())

print("Latent recomputed std:", latent_new.astype(np.float32).std())
print("Latent recomputed mean:", latent_new.astype(np.float32).mean())

tokens = tokenizer([caption]).to(device)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):

    clip_new = get_clip_hidden(clip_model, tokens).cpu().numpy()[0]

print(
    "Latent MAE:",
    np.mean(np.abs(latent_new.astype(np.float32) - latent.astype(np.float32))),
)
print(
    "CLIP MAE:",
    np.mean(np.abs(clip_new.astype(np.float32) - clip_saved.astype(np.float32))),
)

latent_unscaled = latent.astype(np.float32) * LATENT_STD + LATENT_MEAN
latent_unscaled = torch.from_numpy(latent_unscaled).unsqueeze(0).to(device)

with torch.no_grad():
    recon = vae.decode(latent_unscaled)

recon = (recon[0].cpu().clamp(-1, 1) + 1) / 2
recon = recon.permute(1, 2, 0).numpy()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(recon)
axs[1].set_title("Reconstructed")
axs[1].axis("off")

plt.suptitle(caption)

plt.tight_layout()
plt.savefig("diffusion_verify.png")

print("\nSaved visualization to diffusion_verify.png")
