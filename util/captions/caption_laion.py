import os
import gc
import json
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import open_clip

IMAGE_DIR = Path("./data/laion/images/")
OUT_FILE = "./laion_captioned/meta_tags_clip_refined.jsonl"

IMAGE_SIZE = 128
BATCH_SIZE = 24

QWEN_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading Qwen...")

processor = AutoProcessor.from_pretrained(
    QWEN_MODEL,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
)

processor.tokenizer.padding_side = "left"

model = (
    Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.float16
    )
    .to(device)
    .eval()
)

print("Loading CLIP...")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)

clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model = clip_model.to(device).eval()


def clean_tags(text):

    text = text.lower().strip()

    text = re.sub(r"[.;]", ",", text)
    text = re.sub(r"\s+", " ", text)

    tags = [t.strip() for t in text.split(",")]

    seen = set()
    out = []

    for t in tags:
        if len(t) > 1 and t not in seen:
            seen.add(t)
            out.append(t)

    return ", ".join(out[:6])


@torch.inference_mode()
def clip_scores(images, captions):

    imgs = torch.stack([clip_preprocess(im) for im in images]).to(device)

    txt = clip_tokenizer(captions).to(device)

    img_feat = F.normalize(clip_model.encode_image(imgs), dim=-1)
    txt_feat = F.normalize(clip_model.encode_text(txt), dim=-1)

    sims = (img_feat * txt_feat).sum(dim=-1)

    return sims.cpu().tolist()


@torch.inference_mode()
def generate(images):

    prompts = [
        """Return ONLY comma-separated visual object tags.
           Rules:
           - physical objects
           - nouns only
           - lowercase
           - 3-6 tags
           Tags:"""
        for _ in images
    ]

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
        for img, prompt in zip(images, prompts)
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
        max_new_tokens=32,
        do_sample=True,
        temperature=0.55,
        top_p=0.9,
        top_k=50
    )

    prompt_lens = (inputs.input_ids != processor.tokenizer.pad_token_id).sum(dim=1)

    outputs = []

    for i in range(gen.shape[0]):

        toks = gen[i, prompt_lens[i] :]

        s = processor.tokenizer.decode(toks, skip_special_tokens=True)

        if "assistant" in s:
            s = s.split("assistant")[-1]

        outputs.append(clean_tags(s))

    return outputs


def main():

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    images = sorted(
        [
            p
            for p in IMAGE_DIR.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    print("Images on disk:", len(images))

    done = set()
    index = 0
    if os.path.exists(OUT_FILE):

        with open(OUT_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(Path(rec["image_path"]).name)
                    index += 1
                except:
                    pass

    print("Already captioned:", len(done))

    missing = [p for p in images if p.name not in done]

    print("Missing captions:", len(missing))

    batch_imgs = []
    batch_paths = []

    with open(OUT_FILE, "a") as out_f:
        if len(OUT_FILE) > 0:
            out_f.write("")
        for p in tqdm(missing):

            try:
                img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            except Exception as e:
                print("FAILED:", p, e)
                continue

            batch_imgs.append(img)
            batch_paths.append(p)

            if len(batch_imgs) >= BATCH_SIZE:

                caps = generate(batch_imgs)

                scores = clip_scores(batch_imgs, caps)

                for path, cap, sc in zip(batch_paths, caps, scores):

                    row = {
                        "index": str(index),
                        "image_path": str(path),
                        "caption": cap,
                        "clip_score": float(sc),
                    }

                    out_f.write(json.dumps(row) + "\n")
                    index += 1

                batch_imgs.clear()
                batch_paths.clear()

                gc.collect()

    print("Finished caption recovery.")


if __name__ == "__main__":
    main()
