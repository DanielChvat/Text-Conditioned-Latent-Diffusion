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

IMAGE_DIR = Path("./data/imagenet100/images/")
LABELS_FILE = Path("./data/imagenet100/Labels.json")
OUT_FILE = "./IN100_Captioned/meta_tags_clip_refined_bestofN.jsonl"

IMAGE_SIZE = 128
BATCH_SIZE = 24

NUM_SAMPLES_FIRST_PASS = 3
NUM_SAMPLES_RETRY_PASS = 5

CLIP_THRESHOLD = 0.30
MAX_RETRIES = 2
SAVE_CLIP_SCORE = True

MAX_NEW_TOKENS = 28

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

print("Loading labels...")
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    SYNSET_TO_NAME = {k: v.split(",")[0].strip() for k, v in json.load(f).items()}

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


def normalize_tag_count(
    tags: str, min_tags: int = MIN_TAGS, max_tags: int = MAX_TAGS
) -> str:
    parts = [p.strip() for p in tags.split(",") if p.strip()]
    if len(parts) > max_tags:
        parts = parts[:max_tags]
    return ", ".join(parts)


@torch.inference_mode()
def clip_scores(images_pil: List[Image.Image], captions: List[str]) -> List[float]:
    imgs = torch.stack([clip_preprocess(im) for im in images_pil], dim=0).to(device)
    txt = clip_tokenizer(captions).to(device)

    img_feat = clip_model.encode_image(imgs)
    txt_feat = clip_model.encode_text(txt)

    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)

    sims = (img_feat * txt_feat).sum(dim=-1)
    return sims.detach().float().cpu().tolist()


@torch.inference_mode()
def clip_scores_grouped(
    images_pil: List[Image.Image],
    captions_grouped: List[List[str]],
) -> List[List[float]]:
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
    for i in range(B):
        scores_grouped.append(flat_scores[idx : idx + N])
        idx += N
    return scores_grouped


def build_primary_prompts(class_names):
    return [
        f"""Return ONLY comma-separated visual object tags.

            Strict rules:
            - physical objects only
            - things that can be pointed at
            - NO activities
            - NO emotions
            - NO roles (no fisherman, angler, person smiling)
            - nouns only
            - lowercase
            - {MIN_TAGS}-{MAX_TAGS} tags
            - include "{cls}" only if visible

            Output format:
            object, object, object

            Tags:"""
        for cls in class_names
    ]


def build_retry_prompts(class_names: List[str]) -> List[str]:
    variants = [
        "Describe only visible objects.",
        "Focus strictly on physical items present.",
        "List concrete visual entities only.",
        "Return alternative visible tags emphasizing different objects.",
        "Avoid guessing; omit anything uncertain.",
    ]
    return [
        f"""The previous tags mismatched the image.
        {random.choice(variants)}
        Return NEW comma-separated visual tags.
        Rules:
        - only directly visible physical things
        - noun phrases only, lowercase
        - {MIN_TAGS} to {MAX_TAGS} tags
        - include "{cls}" ONLY if it is clearly visible; otherwise omit it
        - no abstract words, no activities, no emotions

        Tags:"""
        for cls in class_names
    ]


@torch.inference_mode()
def qwen_generate_tags_bestofN(
    images_pil: List[Image.Image],
    prompts: List[str],
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[List[str], List[List[str]]]:

    assert len(images_pil) == len(prompts)
    B = len(images_pil)

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
        s = qwen_processor.decode(toks, skip_special_tokens=True).strip()
        s = qwen_processor.tokenizer.decode(
            toks,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if "assistant" in s:
            s = s.split("assistant", 1)[-1]

        s = s.strip()
        s = normalize_tag_count(clean_tags(s), MIN_TAGS, MAX_TAGS)
        decoded.append(s)

    candidates_grouped = []
    idx = 0
    for _ in range(B):
        candidates_grouped.append(decoded[idx : idx + num_samples])
        idx += num_samples

    del inputs, gen, image_inputs, video_inputs, messages, text
    torch.cuda.empty_cache()
    gc.collect()

    best_tags = [cands[0] for cands in candidates_grouped]
    return best_tags, candidates_grouped


def pick_best_by_clip(
    images_pil: List[Image.Image],
    candidates_grouped: List[List[str]],
) -> Tuple[List[str], List[float]]:
    scores_grouped = clip_scores_grouped(images_pil, candidates_grouped)
    best_caps = []
    best_scores = []
    for cands, scores in zip(candidates_grouped, scores_grouped):
        j = max(range(len(scores)), key=lambda k: scores[k])
        best_caps.append(cands[j])
        best_scores.append(scores[j])
    return best_caps, best_scores


def refine_batch_bestofN(
    images_pil: List[Image.Image],
    class_names: List[str],
) -> Tuple[List[str], List[float], List[int]]:
    n = len(images_pil)
    attempts = [0] * n

    _, cand0 = qwen_generate_tags_bestofN(
        images_pil=images_pil,
        prompts=build_primary_prompts(class_names),
        num_samples=NUM_SAMPLES_FIRST_PASS,
        temperature=TEMP_FIRST,
        top_p=TOP_P_FIRST,
        top_k=TOP_K_FIRST,
    )
    captions, scores = pick_best_by_clip(images_pil, cand0)

    for r in range(1, MAX_RETRIES + 1):
        bad_idx = [i for i, s in enumerate(scores) if s < CLIP_THRESHOLD]
        if not bad_idx:
            break

        bad_imgs = [images_pil[i] for i in bad_idx]
        bad_classes = [class_names[i] for i in bad_idx]

        temp = TEMP_RETRY_BASE + 0.20 * (r - 1)
        top_p = max(0.60, TOP_P_RETRY_BASE - 0.08 * (r - 1))
        top_k = TOP_K_RETRY_BASE + 40 * (r - 1)

        _, cand_r = qwen_generate_tags_bestofN(
            images_pil=bad_imgs,
            prompts=build_retry_prompts(bad_classes),
            num_samples=NUM_SAMPLES_RETRY_PASS,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
        )
        new_caps, new_scores = pick_best_by_clip(bad_imgs, cand_r)

        for j, i in enumerate(bad_idx):
            if new_scores[j] > scores[i]:
                captions[i] = new_caps[j]
                scores[i] = new_scores[j]
                attempts[i] = r

    return captions, scores, attempts


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

    batch_paths, batch_images, batch_classes, batch_synsets = [], [], [], []
    global_index = 0

    kept = 0
    rescued = 0
    still_low = 0

    with open(OUT_FILE, "a", buffering=1, encoding="utf-8") as f:
        for p in tqdm(paths):
            synset = p.parent.name
            class_name = SYNSET_TO_NAME.get(synset, synset.replace("_", " "))

            try:
                img = (
                    Image.open(p)
                    .convert("RGB")
                    .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
                )
            except Exception as e:
                print(f"FAILED load: {p} — {e}")
                continue

            batch_paths.append(p)
            batch_images.append(img)
            batch_classes.append(class_name)
            batch_synsets.append(synset)

            if len(batch_paths) >= BATCH_SIZE:
                captions, scores, attempts = refine_batch_bestofN(
                    batch_images, batch_classes
                )

                for i, (path, cls, syn, cap, sc, att) in enumerate(
                    zip(
                        batch_paths,
                        batch_classes,
                        batch_synsets,
                        captions,
                        scores,
                        attempts,
                    )
                ):
                    row = {
                        "index": global_index + i,
                        "image_path": str(path),
                        "synset": syn,
                        "class_name": cls,
                        "caption": cap,
                    }
                    if SAVE_CLIP_SCORE:
                        row["clip_score"] = float(sc)
                        row["regen_attempts"] = int(att)

                    f.write(json.dumps(row) + "\n")
                    kept += 1
                    if att > 0:
                        rescued += 1
                    if sc < CLIP_THRESHOLD:
                        still_low += 1

                global_index += len(batch_paths)

                batch_paths.clear()
                batch_images.clear()
                batch_classes.clear()
                batch_synsets.clear()
                gc.collect()

        if batch_paths:
            captions, scores, attempts = refine_batch_bestofN(
                batch_images, batch_classes
            )

            for i, (path, cls, syn, cap, sc, att) in enumerate(
                zip(
                    batch_paths,
                    batch_classes,
                    batch_synsets,
                    captions,
                    scores,
                    attempts,
                )
            ):
                row = {
                    "index": global_index + i,
                    "image_path": str(path),
                    "synset": syn,
                    "class_name": cls,
                    "caption": cap,
                }
                if SAVE_CLIP_SCORE:
                    row["clip_score"] = float(sc)
                    row["regen_attempts"] = int(att)

                f.write(json.dumps(row) + "\n")
                kept += 1
                if att > 0:
                    rescued += 1
                if sc < CLIP_THRESHOLD:
                    still_low += 1

    print(f"Done. Saved to {OUT_FILE}")
    print(f"Total saved: {kept}")
    print(f"Regenerated/improved: {rescued}")
    print(f"Still below threshold (kept anyway): {still_low}")


if __name__ == "__main__":
    main()
