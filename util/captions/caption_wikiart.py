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

IMAGE_DIR = Path("./data/wikiart/images/")
METADATA_FILE = Path("./data/wikiart/metadata.jsonl")
OUT_FILE = "./wikiart_captioned/meta_tags_clip_refined_bestofN.jsonl"

IMAGE_SIZE = 128
BATCH_SIZE = 24

NUM_SAMPLES_FIRST_PASS = 3
NUM_SAMPLES_RETRY_PASS = 5

CLIP_THRESHOLD = 0.28
MAX_RETRIES = 2
SAVE_CLIP_SCORE = True

MAX_NEW_TOKENS = 32

TEMP_FIRST = 0.55
TOP_P_FIRST = 0.90
TOP_K_FIRST = 50

TEMP_RETRY_BASE = 0.75
TOP_P_RETRY_BASE = 0.85
TOP_K_RETRY_BASE = 80

MIN_TAGS = 4
MAX_TAGS = 8
MAX_TAGS_AFTER_CLEAN = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

print("Loading WikiArt metadata...")
metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line.strip()))

print(f"Loaded {len(metadata)} entries.")

print("Loading Qwen...")
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

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME,
    pretrained=CLIP_PRETRAINED,
)
clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
clip_model = clip_model.to(device).eval()


def extract_title(original_path: str) -> str:
    fname = Path(original_path).stem
    if "_" in fname:
        title = fname.split("_", 1)[1]
    else:
        title = fname
    return title.replace("-", " ")


def clean_tags(text: str, max_tags: int = MAX_TAGS_AFTER_CLEAN) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.;]", ",", text)
    text = re.sub(r"\s+", " ", text)

    tags = [t.strip() for t in text.split(",") if len(t.strip()) > 1]

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
def clip_scores(images_pil, captions):
    imgs = torch.stack([clip_preprocess(im) for im in images_pil]).to(device)
    txt = clip_tokenizer(captions).to(device)

    img_feat = F.normalize(clip_model.encode_image(imgs), dim=-1)
    txt_feat = F.normalize(clip_model.encode_text(txt), dim=-1)

    sims = (img_feat * txt_feat).sum(dim=-1)
    return sims.detach().cpu().tolist()


@torch.inference_mode()
def clip_scores_grouped(images_pil, captions_grouped):
    flat_imgs, flat_txt = [], []
    for i in range(len(images_pil)):
        for cap in captions_grouped[i]:
            flat_imgs.append(images_pil[i])
            flat_txt.append(cap)

    flat_scores = clip_scores(flat_imgs, flat_txt)

    grouped = []
    idx = 0
    for group in captions_grouped:
        grouped.append(flat_scores[idx : idx + len(group)])
        idx += len(group)
    return grouped


def build_primary_prompts(titles, artists, genres):
    prompts = []

    for t, a, g in zip(titles, artists, genres):

        prompts.append(
            f"""
            You are generating training tags for a diffusion model.

            Return ONLY a comma-separated list of visible physical elements in the image.

            STRICT REQUIREMENTS:

            Include:
            - objects (cross, sword, chair, book, tree)
            - materials (oil paint, marble, stone, wood, gold leaf)
            - clothing (robe, armor, cloak, sandals)
            - architecture (arch, column, altar, throne, church)
            - landscape (mountain, sky, river, clouds)
            - medium if visible (fresco, oil painting, sculpture, drawing)

            DO NOT include:
            - abstract words (art, artwork, painting, religious, spirituality)
            - interpretation (symbolism, divine, sacred, holiness)
            - style labels (renaissance art, classical art)
            - metadata words (artist, genre, title)
            - emotions
            - actions (preaching, blessing, carrying)
            - years or numbers
            - full sentences

            GOOD example:
            marble statue, nude male figure, contrapposto pose, curly hair, stone pedestal

            GOOD example:
            oil painting, halo, gold background, robe, crown, altar, church interior

            BAD example:
            religious art, high renaissance, masterpiece, symbolism, emotion

            BAD example:
            this painting shows jesus carrying the cross

            Return {MIN_TAGS}-{MAX_TAGS} tags.
            Lowercase only.
            Comma separated.
            No extra words.

            painting title: {t}
            artist: {a}
            genre: {g}
        """
        )

    return prompts


def build_retry_prompts(titles, artists, genres):

    variants = [
        "Focus on background objects and environment.",
        "Emphasize materials and physical textures.",
        "List overlooked architectural or landscape elements.",
        "Avoid repeating generic religious words.",
    ]

    prompts = []

    for t, a, g in zip(titles, artists, genres):

        prompts.append(
            f"""
            The previous tags were too abstract or mismatched.

            {random.choice(variants)}

            Generate NEW comma-separated physical object tags.

            STRICT RULES:
            Only visible tangible things.
            Nouns only.
            No abstract words.
            No style commentary.
            No artist or genre words unless physically visible.
            No repetition of generic words like:
            art, painting, religious, renaissance, artwork

            GOOD:
            marble, statue, muscular torso, curly hair, pedestal

            GOOD:
            halo, cross, robe, altar, candle, stone wall

            BAD:
            religious art, renaissance painting, spiritual scene

            Return {MIN_TAGS}-{MAX_TAGS} tags.
            Lowercase only.
            Comma separated.
            No explanation.

            painting title: {t}
            artist: {a}
            genre: {g}
        """
        )

    return prompts


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

        if "tags:" in s.lower():
            s = s.lower().split("tags:")[-1]

        if "assistant" in s.lower():
            s = s.lower().split("assistant")[-1]

        for bad_prefix in [
            "painting title:",
            "title:",
            "artist:",
            "genre:",
        ]:
            if bad_prefix in s.lower():
                s = s.lower().split(bad_prefix)[-1]

        s = re.sub(r"^[^a-z]+", "", s.lower())

        s = re.sub(r"[^a-z0-9,\s\-]", "", s)

        s = normalize_tag_count(clean_tags(s))

        decoded.append(s)

    grouped = []
    idx = 0
    for _ in range(len(images_pil)):
        grouped.append(decoded[idx : idx + num_samples])
        idx += num_samples

    del inputs, gen, image_inputs, video_inputs
    torch.cuda.empty_cache()
    gc.collect()

    return grouped


def pick_best_by_clip(images_pil, candidates_grouped):
    scores_grouped = clip_scores_grouped(images_pil, candidates_grouped)
    best_caps, best_scores = [], []
    for cands, scores in zip(candidates_grouped, scores_grouped):
        j = max(range(len(scores)), key=lambda k: scores[k])
        best_caps.append(cands[j])
        best_scores.append(scores[j])
    return best_caps, best_scores


def refine_batch(images, titles, artists, genres):

    cand0 = qwen_generate_tags_bestofN(
        images,
        build_primary_prompts(titles, artists, genres),
        NUM_SAMPLES_FIRST_PASS,
        TEMP_FIRST,
        TOP_P_FIRST,
        TOP_K_FIRST,
    )

    captions, scores = pick_best_by_clip(images, cand0)
    attempts = [0] * len(images)

    for r in range(1, MAX_RETRIES + 1):
        bad_idx = [i for i, s in enumerate(scores) if s < CLIP_THRESHOLD]
        if not bad_idx:
            break

        bad_imgs = [images[i] for i in bad_idx]
        bad_titles = [titles[i] for i in bad_idx]
        bad_artists = [artists[i] for i in bad_idx]
        bad_genres = [genres[i] for i in bad_idx]

        cand_r = qwen_generate_tags_bestofN(
            bad_imgs,
            build_retry_prompts(bad_titles, bad_artists, bad_genres),
            NUM_SAMPLES_RETRY_PASS,
            TEMP_RETRY_BASE,
            TOP_P_RETRY_BASE,
            TOP_K_RETRY_BASE,
        )

        new_caps, new_scores = pick_best_by_clip(bad_imgs, cand_r)

        for j, i in enumerate(bad_idx):
            if new_scores[j] > scores[i]:
                captions[i] = new_caps[j]
                scores[i] = new_scores[j]
                attempts[i] = r

    return captions, scores, attempts


def main():

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    batch_imgs, batch_titles, batch_artists, batch_genres, batch_entries = (
        [],
        [],
        [],
        [],
        [],
    )

    kept = 0
    rescued = 0
    still_low = 0

    with open(OUT_FILE, "w", encoding="utf-8") as f:

        for entry in tqdm(metadata):

            img_path = IMAGE_DIR / entry["file_name"]

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            except:
                continue

            title = extract_title(entry["original_path"])

            batch_imgs.append(img)
            batch_titles.append(title)
            batch_artists.append(entry["artist"])
            batch_genres.append(entry["genre"])
            batch_entries.append(entry)

            if len(batch_imgs) >= BATCH_SIZE:

                captions, scores, attempts = refine_batch(
                    batch_imgs, batch_titles, batch_artists, batch_genres
                )

                for e, cap, sc, att in zip(batch_entries, captions, scores, attempts):

                    row = {
                        "file_name": e["file_name"],
                        "artist": e["artist"],
                        "genre": e["genre"],
                        "title": extract_title(e["original_path"]),
                        "original_path": e["original_path"],
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

                batch_imgs.clear()
                batch_titles.clear()
                batch_artists.clear()
                batch_genres.clear()
                batch_entries.clear()
                gc.collect()

    print(f"Done. Saved {kept}")
    print(f"Rescued: {rescued}")
    print(f"Still low: {still_low}")


if __name__ == "__main__":
    main()
