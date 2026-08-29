"""
Build a HELD-OUT probe pool from VQA v2 val, disjoint from the current probe.

WHY VQA v2 RATHER THAN MORE UNK-VQA
  Volume is the lesser reason. VQA v2 ships **10 annotator answers per question**
  plus an `answer_type` label, and both are quality levers the current source
  cannot offer:

    * agreement >= 8/10 removes ambiguous and subjective questions before they
      reach the probe. Annotator noise is exactly what destroyed UNK-VQA's
      control set (one annotator, no agreement signal, 20% of golds drawn from a
      random distractor slot).
    * answer_type == "other" drops yes/no and counting questions wholesale.
      Masking a dog does not make "Is there a dog?" unanswerable: it makes the
      answer "no". The probe's logic requires open questions.

HELD OUT AT THE IMAGE LEVEL
  Every COCO image already used by the current probe is excluded, not merely
  every question. Two questions about the same image share its scene statistics,
  so question-level disjointness would not give an independent replication.

PARAMETERS ARE FROZEN
  Geometry thresholds are copied from the run that produced the existing result
  and must NOT be retuned here. Retuning against the held-out set converts a
  replication into a second exploratory run, which is the whole thing this is
  meant to avoid.

  python vqa2_setup.py
"""
import argparse
import collections
import io
import json
import os
import re
import zipfile

import requests

from coco_setup import resolve_category

HERE = os.path.dirname(os.path.abspath(__file__))
Q_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
A_URL = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"

# FROZEN: identical to the run behind the held-out result. Retuning here would
# convert a replication into a second exploratory run.
MIN_AREA_FRAC = 0.01
MAX_AREA_FRAC = 0.40
MIN_AGREEMENT = 8  # of 10 annotators

EITHER_OR = re.compile(r"\b\w+ or \w+", re.I)


def fetch(url, member, dest):
    if os.path.exists(dest):
        print(f"[vqa2] cached {os.path.basename(dest)}")
        return dest
    print(f"[vqa2] downloading {os.path.basename(url)} ...")
    r = requests.get(url, timeout=1800)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = member if member in zf.namelist() else next(
            n for n in zf.namelist() if n.endswith(".json"))
        with zf.open(name) as src, open(dest, "wb") as out:
            out.write(src.read())
    print(f"[vqa2] wrote {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-dir", default=os.path.join(HERE, "dataset", "coco"))
    p.add_argument("--exclude", default=os.path.join(HERE, "dataset", "probe",
                                                     "manifest.json"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "vqa2_match.json"))
    p.add_argument("--limit", type=int, default=900)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.coco_dir, exist_ok=True)
    q_path = fetch(Q_URL, "v2_OpenEnded_mscoco_val2014_questions.json",
                   os.path.join(args.coco_dir, "vqa2_val_questions.json"))
    a_path = fetch(A_URL, "v2_mscoco_val2014_annotations.json",
                   os.path.join(args.coco_dir, "vqa2_val_annotations.json"))

    questions = {q["question_id"]: q
                 for q in json.load(open(q_path, encoding="utf-8"))["questions"]}
    annots = json.load(open(a_path, encoding="utf-8"))["annotations"]
    print(f"[vqa2] {len(questions)} questions, {len(annots)} annotations")

    coco = json.load(open(os.path.join(args.coco_dir, "instances_val2014.json"),
                          encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cat_names = set(cats.values())
    images = {im["id"]: im for im in coco["images"]}
    per_image = collections.defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd") or not isinstance(a.get("segmentation"), list):
            continue
        if a["segmentation"]:
            per_image[a["image_id"]].append(a)

    used_images = set()
    if os.path.exists(args.exclude):
        for m in json.load(open(args.exclude, encoding="utf-8")):
            used_images.add(m["source_image"])
    print(f"[vqa2] holding out {len(used_images)} images already used")

    stats, matched = collections.Counter(), {}
    for ann in annots:
        img_id = ann["image_id"]
        im = images.get(img_id)
        if im is None:
            stats["no_coco_image"] += 1
            continue
        if im["file_name"] in used_images:
            stats["held_out_overlap"] += 1
            continue
        if ann.get("answer_type") != "other":
            stats["answer_type_not_other"] += 1
            continue
        gold = ann["multiple_choice_answer"]
        agree = sum(1 for a in ann["answers"]
                    if a["answer"].strip().casefold() == gold.strip().casefold())
        if agree < MIN_AGREEMENT:
            stats["low_annotator_agreement"] += 1
            continue
        q = questions.get(ann["question_id"])
        if q is None:
            stats["question_missing"] += 1
            continue
        qt = q["question"]
        if EITHER_OR.search(qt):
            stats["either_or_question"] += 1
            continue
        cat = resolve_category(gold, cat_names)
        if cat is None:
            stats["answer_not_a_category"] += 1
            continue
        head = cat.split()[0]
        if re.search(r"\b" + re.escape(head) + r"s?\b", qt, re.I):
            stats["question_names_target"] += 1
            continue

        area_img = im["width"] * im["height"]
        targets = [a for a in per_image.get(img_id, []) if cats[a["category_id"]] == cat]
        if not targets:
            stats["category_absent"] += 1
            continue
        t_area = sum(a["area"] for a in targets)
        frac = t_area / area_img
        if frac < MIN_AREA_FRAC:
            stats["target_too_small"] += 1
            continue
        if frac > MAX_AREA_FRAC:
            stats["target_too_large"] += 1
            continue
        others = [a for a in per_image[img_id] if cats[a["category_id"]] != cat]
        if not others:
            stats["no_control_object"] += 1
            continue
        best = min(others, key=lambda a: abs(a["area"] - t_area))
        if best["area"] / area_img < MIN_AREA_FRAC * 0.5:
            stats["control_too_small"] += 1
            continue

        stats["MATCHED"] += 1
        key = f"v2{ann['question_id']}:{im['file_name']}"
        matched[key] = {
            "split": "vqa2_heldout", "question": qt, "answer": gold,
            "image_file": im["file_name"], "coco_image_id": img_id,
            "width": im["width"], "height": im["height"],
            "target_category": cat,
            "target_ann_ids": [a["id"] for a in targets],
            "target_area_frac": frac,
            "control_category": cats[best["category_id"]],
            "control_ann_id": best["id"],
            "control_area_frac": best["area"] / area_img,
            "annotator_agreement": agree,
        }

    print("\nfeasibility:")
    for k, v in stats.most_common():
        print(f"  {k:28} {v}")

    # One question per image, so the held-out set has no repeated scenes.
    by_image, kept = {}, {}
    for k, v in matched.items():
        by_image.setdefault(v["image_file"], []).append((k, v))
    import random
    rng = random.Random(args.seed)
    imgs = sorted(by_image)
    rng.shuffle(imgs)
    for f in imgs:
        k, v = max(by_image[f], key=lambda kv: kv[1]["annotator_agreement"])
        kept[k] = v
        if args.limit and len(kept) >= args.limit:
            break
    print(f"\n  matched={len(matched)} across {len(by_image)} unique images")
    print(f"  kept {len(kept)} (one question per image, highest agreement)")
    if kept:
        ag = collections.Counter(v["annotator_agreement"] for v in kept.values())
        print(f"  agreement distribution: {dict(sorted(ag.items()))}")
        top = collections.Counter(v["target_category"] for v in kept.values())
        print(f"  top categories: {dict(top.most_common(8))}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(kept, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
