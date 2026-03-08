import os
import math
import time
import argparse
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torchvision import transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm
import lpips

from dataset.vae_dataset import get_vae_dataloader
from model.vae.vae import VAE
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
from itertools import islice

from util.edge_detector import EdgeDetector
from util.LBP import LBPLoss

torch.backends.cudnn.benchmark = True

sample_dir = "vae_training_samples"
ckpt_dir = "checkpoints"

epochs = 4
batch_size = 2

BASE_LR = 3e-6
FULL_DATASET_STEPS_PER_EPOCH = math.ceil(284000 / batch_size)

optimizer_beta1 = 0.9
optimizer_beta2 = 0.999
weight_decay = 1e-4

LPIPS_WEIGHT_START = 0.2
LPIPS_WEIGHT_END = 1.0

USE_AMP = True
INPUT_RES = 512
SAVES_PER_EPOCH = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def save_reconstructions(model, batch, step):
    os.makedirs(sample_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        recon, _, _, _ = model(batch.to(device))

    x = batch[:4]
    y = recon[:4]

    grid = vutils.make_grid(
        torch.cat([(x + 1) / 2, (y + 1) / 2], dim=0),
        nrow=2
    )

    vutils.save_image(grid, os.path.join(sample_dir, f"recon_{step}.png"))

    model.train()


def save_checkpoint(path, state, keep_last=2):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

    ckpt_dir = os.path.dirname(path)

    ckpts = sorted(
        [
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir)
            if f.endswith(".pt")
        ],
        key=os.path.getmtime
    )

    while len(ckpts) > keep_last:
        os.remove(ckpts.pop(0))


parser = argparse.ArgumentParser()

parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--reset_opt", action="store_true")
parser.add_argument("--subset", type=int, default=None)
parser.add_argument("--run_tag", type=str, default=None)

args = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize(INPUT_RES + 64),
    transforms.RandomCrop(INPUT_RES),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


loader_full = get_vae_dataloader(
    roots=[
        "./data/imagenet100",
        "./data/coco",
        "./data/laion",
        "./data/wikiart",
        "./data/stanford_cars",
    ],
    batch_size=batch_size,
    transform=transform,
)


if args.subset:
    idx = torch.randperm(len(loader_full.dataset))[:args.subset]
    dataset = torch.utils.data.Subset(loader_full.dataset, idx)
else:
    dataset = loader_full.dataset


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
)


steps_per_epoch = len(loader)
total_steps = epochs * steps_per_epoch
save_every = max(1, steps_per_epoch // SAVES_PER_EPOCH)

lr_scale = math.sqrt(FULL_DATASET_STEPS_PER_EPOCH / max(1, steps_per_epoch))
lr_scale = min(lr_scale, 12.0)
lr = BASE_LR * lr_scale


vae = VAE(
    z_ch=16,
    base_ch=128,
    ch_mult=(1,2,4),
    enc_num_res=4,
    dec_num_res=8,
    attn_resolutions=(64,128),
    input_res=INPUT_RES,
).to(device)


ema = ExponentialMovingAverage(vae.parameters(), decay=0.999)


edge_detector = EdgeDetector().to(device).eval()
for p in edge_detector.parameters():
    p.requires_grad_(False)


lbp_loss_fn = LBPLoss(tau=10).to(device)
for p in lbp_loss_fn.parameters():
    p.requires_grad_(False)


optimizer = torch.optim.AdamW(
    vae.parameters(),
    lr=lr,
    betas=(optimizer_beta1, optimizer_beta2),
    weight_decay=weight_decay,
)


scaler = GradScaler(enabled=USE_AMP)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=lr * 0.1,
)


global_step = 0
start_epoch = 0
start_step_in_epoch = 0


if args.resume:

    ckpt = torch.load(args.resume, map_location=device)

    vae.load_state_dict(ckpt["vae"], strict=False)
    ema.load_state_dict(ckpt["ema"])

    global_step = ckpt["global_step"]

    start_epoch = global_step // steps_per_epoch
    start_step_in_epoch = global_step % steps_per_epoch

    if not args.reset_opt:
        optimizer.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])


print(
    f"[resume] global_step={global_step} "
    f"start_epoch={start_epoch} "
    f"skip_steps={start_step_in_epoch}"
)


lpips_loss = lpips.LPIPS(net="alex").to(device).eval()


run_tag = args.run_tag or f"run_{int(time.time())}"
writer = SummaryWriter(f"runs/vae_{run_tag}")


for epoch in range(start_epoch, epochs):

    vae.train()

    initial = start_step_in_epoch if epoch == start_epoch else 0

    if epoch == start_epoch:
        epoch_loader = islice(loader, start_step_in_epoch, None)
    else:
        epoch_loader = loader

    pbar = tqdm(epoch_loader, ncols=120, initial=initial, total=steps_per_epoch)

    for images in pbar:

        images = images.to(device, non_blocking=True)

        global_step += 1

        with autocast(device_type="cuda", enabled=USE_AMP):
            recon, mu, logvar, _ = vae(images)

        recon_f  = recon.float()
        images_f = images.float()

        _, edges_real = edge_detector(images_f.detach())
        _, edges_fake = edge_detector(recon_f)

        rec_l1  = F.l1_loss(recon_f, images_f)
        rec_mse = F.mse_loss(recon_f, images_f)

        rec_loss = 0.95 * rec_l1 + 0.05 * rec_mse

        edge_loss = F.l1_loss(edges_fake, edges_real)

        texture_loss = lbp_loss_fn(recon_f, images_f)

        with autocast(device_type="cuda", enabled=False):
            lp_loss = lpips_loss(
                recon.float().clamp(-1,1),
                images.float()
            ).mean()

        lp_ramp = min(1.0, global_step / max(1, steps_per_epoch))
        lp_weight = LPIPS_WEIGHT_START + lp_ramp * (LPIPS_WEIGHT_END - LPIPS_WEIGHT_START)

        kl_loss = kl_divergence(mu, logvar)

        kl_weight = 0.05

        loss = (
            rec_loss
            + lp_weight * lp_loss
            + kl_weight * kl_loss
            + 0.05 * edge_loss
            + 0.1 * texture_loss
        )

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

        scale_before = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()

        if scaler.get_scale() >= scale_before:
            ema.update()
            scheduler.step()

        pbar.set_description(
            f"E{epoch+1} G{global_step} "
            f"r:{rec_loss.item():.3f} "
            f"p:{lp_loss.item():.3f} "
            f"kl:{kl_loss.item():.3f} "
            f"e:{edge_loss.item():.3f} "
            f"tex:{texture_loss.item():.3f}"
        )

        if global_step % save_every == 0:

            with ema.average_parameters():
                save_reconstructions(vae, images, global_step)

            writer.add_scalar("loss/rec", rec_loss.item(), global_step)
            writer.add_scalar("loss/lpips", lp_loss.item(), global_step)
            writer.add_scalar("loss/kl", kl_loss.item(), global_step)
            writer.add_scalar("loss/edge", edge_loss.item(), global_step)
            writer.add_scalar("loss/texture", texture_loss.item(), global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            ckpt_path = f"{ckpt_dir}/vae_step_{global_step}.pt"

            state = {
                "vae": vae.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "global_step": global_step,
            }

            save_checkpoint(ckpt_path, state)


    ckpt_path = f"{ckpt_dir}/vae_epoch_{epoch}.pt"

    state = {
        "vae": vae.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "global_step": global_step,
    }

    save_checkpoint(ckpt_path)

    start_step_in_epoch = 0


writer.close()