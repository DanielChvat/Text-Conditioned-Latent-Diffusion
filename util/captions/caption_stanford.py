import os
import gc
import json
import re
import random
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import open_clip

QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

IMAGE_DIR = Path("./data/stanford_cars/images/")
METADATA_FILE = "./data/stanford_cars/metadata.jsonl"
OUT_FILE = "./Stanford_Cars_Captioned/meta_tags_clip_refined.jsonl"

IMAGE_SIZE = 128
BATCH_SIZE = 24

NUM_SAMPLES_FIRST_PASS = 3

CLIP_THRESHOLD = 0.30
SAVE_CLIP_SCORE = True

MAX_NEW_TOKENS = 32

TEMP_FIRST = 0.55
TOP_P_FIRST = 0.90
TOP_K_FIRST = 50

MIN_TAGS = 3
MAX_TAGS = 6
MAX_TAGS_AFTER_CLEAN = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

print("Loading Qwen2-VL...")

processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_NAME,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
)

processor.tokenizer.padding_side = "left"

model = (
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


def clean_tags(text: str):

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

    return ", ".join(out[:MAX_TAGS_AFTER_CLEAN])


def normalize_tag_count(tags: str):

    parts = [p.strip() for p in tags.split(",") if p.strip()]

    if len(parts) > MAX_TAGS:
        parts = parts[:MAX_TAGS]

    return ", ".join(parts)


@torch.inference_mode()
def clip_scores(images_pil: List[Image.Image], captions: List[str]):

    imgs = torch.stack([clip_preprocess(im) for im in images_pil], dim=0).to(device)

    txt = clip_tokenizer(captions).to(device)

    img_feat = F.normalize(clip_model.encode_image(imgs), dim=-1)
    txt_feat = F.normalize(clip_model.encode_text(txt), dim=-1)

    sims = (img_feat * txt_feat).sum(dim=-1)

    return sims.detach().cpu().tolist()


@torch.inference_mode()
def clip_scores_grouped(images_pil, captions_grouped):

    B = len(images_pil)
    N = len(captions_grouped[0]) if B > 0 else 0

    flat_imgs = []
    flat_txt = []

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


def build_primary_prompts(class_names):
    prompts = []

    for cls in class_names:
        prompts.append(
            f"""
            Return ONLY comma-separated visual tags.

            IMPORTANT:
            - the FIRST tag MUST be "{cls.lower()}"

            Rules:
            - physical objects only
            - noun phrases only
            - lowercase
            - {MIN_TAGS}-{MAX_TAGS} tags total
            - do not invent models
            - include "{cls.lower()}" exactly once

            Example:
            {cls.lower()}, car, wheel, road

            Tags:
        """
        )

    return prompts


@torch.inference_mode()
def qwen_generate_tags_bestofN(images_pil, prompts, num_samples):

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

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    gen = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMP_FIRST,
        top_p=TOP_P_FIRST,
        top_k=TOP_K_FIRST,
        repetition_penalty=1.05,
        num_return_sequences=num_samples,
    )

    prompt_lens = (
        (inputs.input_ids != processor.tokenizer.pad_token_id).sum(dim=1).tolist()
    )

    decoded = []

    for k in range(gen.shape[0]):

        i = k // num_samples
        L = prompt_lens[i]

        toks = gen[k, L:]

        s = processor.tokenizer.decode(toks, skip_special_tokens=True)

        s = s.strip()

        if "assistant" in s.lower():
            s = re.split(r"assistant[:\s]*", s, flags=re.IGNORECASE)[-1]

        s = s.strip(" \n\"'")

        s = normalize_tag_count(clean_tags(s))

        decoded.append(s)

    candidates_grouped = []

    idx = 0

    for _ in range(len(images_pil)):
        candidates_grouped.append(decoded[idx : idx + num_samples])
        idx += num_samples

    del inputs, gen
    torch.cuda.empty_cache()

    return candidates_grouped


def main():

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    batch_paths = []
    batch_images = []
    batch_classes = []

    global_index = 0

    with open(METADATA_FILE) as meta_f, open(OUT_FILE, "w", buffering=1) as f:

        for line in tqdm(meta_f):

            rec = json.loads(line)

            img_path = IMAGE_DIR / rec["file_name"]

            try:
                img = (
                    Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                )
            except:
                continue

            batch_paths.append(img_path)
            batch_classes.append(rec["label_name"])
            batch_images.append(img)
            if len(batch_paths) >= BATCH_SIZE:

                prompts = build_primary_prompts(batch_classes)

                cand = qwen_generate_tags_bestofN(
                    batch_images, prompts, NUM_SAMPLES_FIRST_PASS
                )

                captions = []
                scores = []

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
                batch_classes.clear()
                batch_images.clear()

                gc.collect()

        if batch_paths:

            prompts = build_primary_prompts(len(batch_images))

            cand = qwen_generate_tags_bestofN(
                batch_images, prompts, NUM_SAMPLES_FIRST_PASS
            )

            captions = []
            scores = []

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
