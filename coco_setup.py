"""
Step 1 of the controlled-probe pipeline: fetch COCO instance annotations and
report how many VQA questions can be aligned to annotated answer objects.

WHY A NEW CONTROLLED PROBE
  Coarse image perturbations can remove evidence from nominal should-answer
  controls, making false-abstention rates difficult to interpret. We instead use
  a within-image control that preserves the annotated answer object.

THE FIX
  COCO ships instance annotations. If a question's answer names a category in
  the image, we mask its padded bounding box and separately mask a different
  object of approximately matched painted area. The complete-target label is an
  operational assumption, not guaranteed ground truth: context can retain answer
  information and annotations can omit instances. The paper and release report
  this limitation and provide control-matching sensitivity analyses.

  Questions and short answers originate from VQA data aligned to COCO images.

  python coco_setup.py
"""
import argparse
import collections
import io
import json
import os
import zipfile

import requests

from unkvqa.data import load_items, load_origin_questions

HERE = os.path.dirname(os.path.abspath(__file__))
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
WANTED_MEMBER = "annotations/instances_val2014.json"

#: VQA answers are colloquial; COCO categories are not. Only unambiguous
#: mappings are listed: a wrong mapping would mask the wrong object and
#: silently reintroduce the label noise this pipeline exists to remove.
SYNONYMS = {
    "man": "person", "woman": "person", "men": "person", "women": "person",
    "boy": "person", "girl": "person", "child": "person", "kid": "person",
    "people": "person", "guy": "person", "lady": "person", "player": "person",
    "skier": "person", "surfer": "person", "rider": "person", "baby": "person",
    "dogs": "dog", "cats": "cat", "cows": "cow", "horses": "horse",
    "sheeps": "sheep", "birds": "bird", "elephants": "elephant",
    "zebras": "zebra", "giraffes": "giraffe", "bears": "bear",
    "puppy": "dog", "kitten": "cat", "cattle": "cow", "calf": "cow",
    "bull": "cow", "pony": "horse", "teddy": "teddy bear",
    "bike": "bicycle", "bikes": "bicycle", "motorbike": "motorcycle",
    "cellphone": "cell phone", "phone": "cell phone", "tv": "tv",
    "television": "tv", "laptops": "laptop", "computer": "laptop",
    "couch": "couch", "sofa": "couch", "plane": "airplane",
    "airplanes": "airplane", "jet": "airplane", "bus": "bus", "buses": "bus",
    "trains": "train", "cars": "car", "trucks": "truck", "boats": "boat",
    "sandwiches": "sandwich", "pizzas": "pizza", "bananas": "banana",
    "apples": "apple", "oranges": "orange", "donut": "donut",
    "doughnut": "donut", "doughnuts": "donut", "donuts": "donut",
    "hotdog": "hot dog", "hot dogs": "hot dog", "broccolli": "broccoli",
    "umbrellas": "umbrella", "kites": "kite", "surfboards": "surfboard",
    "skateboards": "skateboard", "racket": "tennis racket",
    "racquet": "tennis racket", "bat": "baseball bat",
    "glove": "baseball glove", "frisbee": "frisbee",
    "clocks": "clock", "vases": "vase", "chairs": "chair",
    "bottles": "bottle", "cups": "cup", "bowls": "bowl", "plates": "plate",
    "toilets": "toilet", "beds": "bed", "books": "book",
    "flowers": "potted plant", "plant": "potted plant", "plants": "potted plant",
    "signs": "stop sign", "hydrant": "fire hydrant",
    "scissor": "scissors", "luggage": "suitcase", "bag": "handbag",
    "purse": "handbag", "sink": "sink", "oven": "oven", "microwave": "microwave",
}


def fetch_annotations(dest_dir: str) -> str:
    path = os.path.join(dest_dir, "instances_val2014.json")
    if os.path.exists(path):
        print(f"[coco] using cached {path}")
        return path
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[coco] downloading {COCO_ANN_URL} (~241 MB, one time)...")
    r = requests.get(COCO_ANN_URL, timeout=1800, stream=True)
    r.raise_for_status()
    blob = io.BytesIO(r.content)
    with zipfile.ZipFile(blob) as zf:
        with zf.open(WANTED_MEMBER) as src, open(path, "wb") as out:
            out.write(src.read())
    print(f"[coco] wrote {path} ({os.path.getsize(path)/1e6:.0f} MB)")
    return path


def normalize_answer(a: str) -> str:
    a = a.strip().casefold()
    for art in ("a ", "an ", "the "):
        if a.startswith(art):
            a = a[len(art):]
    return a


def resolve_category(answer: str, cat_names: set[str]) -> str | None:
    a = normalize_answer(answer)
    if a in cat_names:
        return a
    if a in SYNONYMS and SYNONYMS[a] in cat_names:
        return SYNONYMS[a]
    if a.endswith("s") and a[:-1] in cat_names:  # simple plural
        return a[:-1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=os.path.join(HERE, "dataset", "UNK-VQA"))
    p.add_argument("--coco-dir", default=os.path.join(HERE, "dataset", "coco"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "coco_match.json"))
    p.add_argument("--min-area-frac", type=float, default=0.01,
                   help="skip target objects smaller than this fraction of the image")
    p.add_argument("--max-area-frac", type=float, default=0.40,
                   help="skip targets larger than this: a huge mask reintroduces "
                        "the very problem we are fixing")
    p.add_argument("--source", default="all-image", choices=["i2", "all-image"],
                   help="'i2' = the 2,446 I-2 items only; 'all-image' = every "
                        "unperturbed question underlying an I-1/I-2/I-3 entry "
                        "(same questions, ~6x the pool)")
    args = p.parse_args()

    ann_path = fetch_annotations(args.coco_dir)
    print("[coco] loading instance annotations...")
    coco = json.load(open(ann_path, encoding="utf-8"))

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_names = set(cats.values())
    images = {im["id"]: im for im in coco["images"]}
    by_file = {im["file_name"]: im["id"] for im in coco["images"]}

    per_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue  # RLE encoding; polygons only, to avoid a pycocotools dep
        if not isinstance(a.get("segmentation"), list) or not a["segmentation"]:
            continue
        per_image[a["image_id"]].append(a)
    print(f"[coco] {len(images)} images, {len(cats)} categories, "
          f"{sum(len(v) for v in per_image.values())} usable polygon instances")

    if args.source == "i2":
        src = [(i.item_id, i.split, i.question, i.gold_original, i.original_file, i.answerable)
               for i in load_items(args.dataset_root, ["val", "test"])]
    else:
        src = [(q.key, q.split, q.question, q.answer, q.image_file, None)
               for q in load_origin_questions(args.dataset_root, ["val", "test"])]
    print(f"[src] {args.source}: {len(src)} candidate question/answer pairs")

    stats = collections.Counter()
    matched = {}

    for item_id, split, question, answer, original_file, answerable in src:
        img_id = by_file.get(original_file)
        if img_id is None:
            stats["no_coco_image"] += 1
            continue
        cat = resolve_category(answer, cat_names)
        if cat is None:
            stats["answer_not_a_category"] += 1
            continue
        im = images[img_id]
        area_img = im["width"] * im["height"]
        targets = [a for a in per_image.get(img_id, []) if cats[a["category_id"]] == cat]
        if not targets:
            stats["category_absent_from_image"] += 1
            continue
        target_area = sum(a["area"] for a in targets)
        frac = target_area / area_img
        if frac < args.min_area_frac:
            stats["target_too_small"] += 1
            continue
        if frac > args.max_area_frac:
            stats["target_too_large"] += 1
            continue
        # A control needs a DIFFERENT object of comparable area to mask instead.
        others = [a for a in per_image[img_id] if cats[a["category_id"]] != cat]
        if not others:
            stats["no_control_object"] += 1
            continue
        best = min(others, key=lambda a: abs(a["area"] - target_area))
        ctrl_frac = best["area"] / area_img
        if ctrl_frac < args.min_area_frac * 0.5:
            stats["control_too_small"] += 1
            continue

        stats["MATCHED"] += 1
        matched[item_id] = {
            "split": split,
            "question": question,
            "answer": answer,
            "image_file": original_file,
            "coco_image_id": img_id,
            "width": im["width"], "height": im["height"],
            "target_category": cat,
            "target_ann_ids": [a["id"] for a in targets],
            "target_area_frac": frac,
            "control_category": cats[best["category_id"]],
            "control_ann_id": best["id"],
            "control_area_frac": ctrl_frac,
        }

    print("\nfeasibility:")
    for k, v in stats.most_common():
        print(f"  {k:28} {v}")
    if stats["MATCHED"]:
        fr = [m["target_area_frac"] for m in matched.values()]
        cf = [m["control_area_frac"] for m in matched.values()]
        fr.sort(); cf.sort()
        print(f"\n  target  area frac: median={fr[len(fr)//2]:.3f} "
              f"min={fr[0]:.3f} max={fr[-1]:.3f}")
        print(f"  control area frac: median={cf[len(cf)//2]:.3f} "
              f"min={cf[0]:.3f} max={cf[-1]:.3f}")
        top = collections.Counter(m["target_category"] for m in matched.values())
        print(f"  top target categories: {dict(top.most_common(8))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(matched, fh, indent=1)
    print(f"\nwrote {args.out}  ({len(matched)} items)")


if __name__ == "__main__":
    main()
