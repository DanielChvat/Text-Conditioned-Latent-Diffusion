import os
import gc
import json
import re
import random
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import open_clip

QWEN_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

IMAGE_DIR = Path("./data/coco/images/")
OUT_FILE = "./coco_captioned/meta_tags_clip_refined_bestofN.jsonl"

IMAGE_SIZE = 128
BATCH_SIZE = 24

NUM_SAMPLES_FIRST_PASS = 3
NUM_SAMPLES_RETRY_PASS = 5

CLIP_THRESHOLD = 0.30
MAX_RETRIES = 2
SAVE_CLIP_SCORE = True

MAX_NEW_TOKENS = 32

TEMP_FIRST = 0.55
TOP_P_FIRST = 0.90
TOP_K_FIRST = 50

TEMP_RETRY_BASE = 0.75
TOP_P_RETRY_BASE = 0.85
TOP_K_RETRY_BASE = 80

MIN_TAGS = 3
MAX_TAGS = 6
MAX_TAGS_AFTER_CLEAN = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading Qwen2-VL...")
qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_NAME,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
)
qwen_processor.tokenizer.padding_side = "left"

qwen_model = (
    Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
    )
    .to(device)
    .eval()
)
print("Qwen ready.")

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME,
    pretrained=CLIP_PRETRAINED,
)
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
clip_model = clip_model.to(device).eval()
print("CLIP ready.")


def clean_tags(text: str, max_tags: int = MAX_TAGS_AFTER_CLEAN) -> str:
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[.;]", ",", text)
    text = re.sub(r"\s+", " ", text)

    tags = [t.strip() for t in text.split(",")]
    tags = [t for t in tags if len(t) > 1]

    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return ", ".join(out[:max_tags])


def normalize_tag_count(tags: str) -> str:
    parts = [p.strip() for p in tags.split(",") if p.strip()]
    if len(parts) > MAX_TAGS:
        parts = parts[:MAX_TAGS]
    return ", ".join(parts)


@torch.inference_mode()
def clip_scores(images_pil: List[Image.Image], captions: List[str]) -> List[float]:
    imgs = torch.stack([clip_preprocess(im) for im in images_pil], dim=0).to(device)
    txt = clip_tokenizer(captions).to(device)

    img_feat = F.normalize(clip_model.encode_image(imgs), dim=-1)
    txt_feat = F.normalize(clip_model.encode_text(txt), dim=-1)

    sims = (img_feat * txt_feat).sum(dim=-1)
    return sims.detach().float().cpu().tolist()


@torch.inference_mode()
def clip_scores_grouped(images_pil, captions_grouped):
    B = len(images_pil)
    N = len(captions_grouped[0]) if B > 0 else 0

    flat_imgs, flat_txt = [], []
    for i in range(B):
        for j in range(N):
            flat_imgs.append(images_pil[i])
            flat_txt.append(captions_grouped[i][j])

    flat_scores = clip_scores(flat_imgs, flat_txt)

    scores_grouped = []
    idx = 0
    for _ in range(B):
        scores_grouped.append(flat_scores[idx : idx + N])
        idx += N
    return scores_grouped


def build_primary_prompts(n: int):
    return [
        f"""
            Return ONLY comma-separated visual object tags.

            Rules:
            - physical objects only
            - things that can be pointed at
            - no activities
            - no emotions
            - noun phrases only
            - lowercase
            - {MIN_TAGS}-{MAX_TAGS} tags

            Output:
            object, object, object

            Tags:"""
        for _ in range(n)
    ]


def build_retry_prompts(n: int):
    variants = [
        "Describe only visible objects.",
        "Focus strictly on concrete items.",
        "List different visible objects.",
        "Avoid guessing.",
    ]
    return [
        f"""
            The previous tags mismatched the image. {random.choice(variants)}

            Return NEW comma-separated object tags.
            - visible physical things only
            - noun phrases only
            - {MIN_TAGS}-{MAX_TAGS} tags
            - lowercase

            Tags:"""
        for _ in range(n)
    ]


@torch.inference_mode()
def qwen_generate_tags_bestofN(
    images_pil, prompts, num_samples, temperature, top_p, top_k
):

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for img, prompt in zip(images_pil, prompts)
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen = qwen_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.05,
        num_return_sequences=num_samples,
    )

    prompt_lens = (
        (inputs.input_ids != qwen_processor.tokenizer.pad_token_id).sum(dim=1).tolist()
    )

    decoded = []
    for k in range(gen.shape[0]):
        i = k // num_samples
        L = prompt_lens[i]
        toks = gen[k, L:]
        s = qwen_processor.decode(toks, skip_special_tokens=True)
        s = normalize_tag_count(clean_tags(s))
        decoded.append(s)

    candidates_grouped = []
    idx = 0
    for _ in range(len(images_pil)):
        candidates_grouped.append(decoded[idx : idx + num_samples])
        idx += num_samples

    del inputs, gen
    torch.cuda.empty_cache()
    gc.collect()

    return candidates_grouped

def main():
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in image_exts])

    print(f"Found {len(paths)} total images")

    done = set()

    if os.path.exists(OUT_FILE):
        print("Loading existing captions for resume...")

        with open(OUT_FILE, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["image_path"])
                except:
                    continue

    print(f"{len(done)} images already captioned")

    paths = [p for p in paths if str(p) not in done]

    print(f"{len(paths)} images remaining to caption")

    print(f"Found {len(paths)} images")
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    batch_paths, batch_images = [], []
    global_index = 0

    with open(OUT_FILE, "a", buffering=1, encoding="utf-8") as f:
        for p in tqdm(paths):
            try:
                img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            except:
                continue

            batch_paths.append(p)
            batch_images.append(img)

            if len(batch_paths) >= BATCH_SIZE:

                prompts = build_primary_prompts(len(batch_images))
                cand = qwen_generate_tags_bestofN(
                    batch_images,
                    prompts,
                    NUM_SAMPLES_FIRST_PASS,
                    TEMP_FIRST,
                    TOP_P_FIRST,
                    TOP_K_FIRST,
                )

                captions, scores = [], []
                grouped_scores = clip_scores_grouped(batch_images, cand)

                for cands, scs in zip(cand, grouped_scores):
                    j = max(range(len(scs)), key=lambda k: scs[k])
                    captions.append(cands[j])
                    scores.append(scs[j])

                for i, (path, cap, sc) in enumerate(zip(batch_paths, captions, scores)):
                    row = {
                        "index": global_index + i,
                        "image_path": str(path),
                        "caption": cap,
                    }
                    if SAVE_CLIP_SCORE:
                        row["clip_score"] = float(sc)
                    f.write(json.dumps(row) + "\n")

                global_index += len(batch_paths)
                batch_paths.clear()
                batch_images.clear()
                gc.collect()

        if batch_paths:
            prompts = build_primary_prompts(len(batch_images))

            cand = qwen_generate_tags_bestofN(
                batch_images,
                prompts,
                NUM_SAMPLES_FIRST_PASS,
                TEMP_FIRST,
                TOP_P_FIRST,
                TOP_K_FIRST,
            )

            captions, scores = [], []
            grouped_scores = clip_scores_grouped(batch_images, cand)

            for cands, scs in zip(cand, grouped_scores):
                j = max(range(len(scs)), key=lambda k: scs[k])
                captions.append(cands[j])
                scores.append(scs[j])

            for i, (path, cap, sc) in enumerate(zip(batch_paths, captions, scores)):
                row = {
                    "index": global_index + i,
                    "image_path": str(path),
                    "caption": cap,
                }

                if SAVE_CLIP_SCORE:
                    row["clip_score"] = float(sc)

                f.write(json.dumps(row) + "\n")
    print(f"Done. Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
