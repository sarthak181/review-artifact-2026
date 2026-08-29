"""Render the graded object-occlusion probe.

The target mask grows from the center of each bounding box. Intermediate target
conditions describe the response curve and are not treated as automatically
unanswerable because they have no dose-matched controls.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

#: Nominal occlusion levels, as a fraction of the target's BOUNDING BOX area.
#:
#: `calibrate_levels.py` selected these values on 150 items to spread measured
#: object-pixel coverage across the response curve:
#:
#:     nominal 11 23 37 54 73 100  ->  true object coverage 20 40 60 80 95 100 %
#:
#: Level 100 remains the final anchor because it guarantees complete bounding-box
#: removal for each item.
LEVELS = (11, 23, 37, 54, 73, 100)


def safe_id(key: str) -> str:
    return key.replace(":", "_").replace(".jpg", "")


def dilate(box, w, h, pad=0.04):
    """COCO xywh box -> padded xyxy, clipped to the image."""
    x, y, bw, bh = box
    px, py = bw * pad, bh * pad
    return [max(0, x - px), max(0, y - py), min(w, x + bw + px), min(h, y + bh + py)]


def scaled_box(box, frac, W, H):
    """Shrink a box toward its centre so it covers `frac` of the original AREA.
    Linear scale = sqrt(area fraction)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = frac ** 0.5
    w, h = (x1 - x0) * s / 2.0, (y1 - y0) * s / 2.0
    return [max(0, cx - w), max(0, cy - h), min(W, cx + w), min(H, cy + h)]


def paint(img, boxes):
    out = img.copy()
    d = ImageDraw.Draw(out)
    cov = Image.new("1", out.size, 0)
    cd = ImageDraw.Draw(cov)
    for b in boxes:
        d.rectangle(b, fill=(0, 0, 0))
        cd.rectangle(b, fill=1)
    return out, sum(cov.getdata()) / (out.width * out.height)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default=os.path.join(HERE, "results", "coco_match.json"))
    p.add_argument("--coco-ann", default=os.path.join(HERE, "dataset", "coco",
                                                      "instances_val2014.json"))
    p.add_argument("--src-images", default=os.path.join(HERE, "dataset", "coco", "val2014"))
    p.add_argument("--base-manifest", default=os.path.join(HERE, "dataset", "probe",
                                                           "manifest.json"))
    p.add_argument("--out", default=os.path.join(HERE, "dataset", "probe_graded"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Reuse the ALREADY-VALIDATED item set: these passed the size and balance
    # filters and the pilot confirmed their full-occlusion masks work. Building a
    # fresh selection would mean re-validating from scratch.
    matched = json.load(open(args.match, encoding="utf-8"))
    by_sid = {safe_id(k): v for k, v in matched.items()}
    if args.base_manifest and os.path.exists(args.base_manifest):
        base = json.load(open(args.base_manifest, encoding="utf-8"))
    else:
        # No pre-validated manifest (a fresh source): fall back to the raw match
        # file. The size/balance filters then have to be applied upstream.
        base = [{"item_id": safe_id(k), "split": v.get("split", "val"),
                 "question": v["question"], "answer": v["answer"],
                 "target_category": v["target_category"],
                 "control_category": v["control_category"],
                 "source_image": v["image_file"]}
                for k, v in matched.items()]

    import random
    random.Random(args.seed).shuffle(base)
    base = base[: args.limit]
    base.sort(key=lambda m: m["item_id"])
    print(f"[graded] {len(base)} items (subset of the validated probe set)")

    coco = json.load(open(args.coco_ann, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    manifest, skipped = [], 0
    for m in base:
        sid = m["item_id"]
        meta = by_sid.get(sid)
        if meta is None:
            skipped += 1
            continue
        src = os.path.join(args.src_images, m["source_image"])
        if not os.path.exists(src):
            skipped += 1
            continue
        img = Image.open(src).convert("RGB")
        W, H = img.size

        t_anns = [ann_by_id[i] for i in meta["target_ann_ids"] if i in ann_by_id]
        c_ann = ann_by_id.get(meta["control_ann_id"])
        if not t_anns or c_ann is None:
            skipped += 1
            continue
        full = [dilate(a["bbox"], W, H) for a in t_anns]

        rec = {"item_id": sid, "split": m.get("split", "val"),
               "question": m["question"], "answer": m["answer"],
               "target_category": m["target_category"],
               "control_category": m["control_category"],
               "source_image": m["source_image"]}

        img.save(os.path.join(img_dir, f"{sid}_orig.jpg"), quality=95)
        rec["image_original"] = f"{sid}_orig.jpg"

        for lvl in LEVELS:
            boxes = [scaled_box(b, lvl / 100.0, W, H) for b in full]
            out, frac = paint(img, boxes)
            name = f"{sid}_t{lvl}.jpg"
            out.save(os.path.join(img_dir, name), quality=95)
            rec[f"image_target_{lvl}"] = name
            rec[f"masked_frac_{lvl}"] = round(frac, 4)

        cimg, cfrac = paint(img, [dilate(c_ann["bbox"], W, H)])
        cimg.save(os.path.join(img_dir, f"{sid}_control.jpg"), quality=95)
        rec["image_control"] = f"{sid}_control.jpg"
        rec["control_masked_frac"] = round(cfrac, 4)
        manifest.append(rec)

    out_path = os.path.join(args.out, "manifest.json")
    json.dump(manifest, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"[graded] rendered {len(manifest)} items ({skipped} skipped) -> {img_dir}")
    for lvl in LEVELS:
        v = sorted(r[f"masked_frac_{lvl}"] for r in manifest)
        print(f"  target_{lvl:3}%  painted frac median={v[len(v)//2]:.3f}")
    v = sorted(r["control_masked_frac"] for r in manifest)
    print(f"  control     painted frac median={v[len(v)//2]:.3f}")
    print(f"[graded] wrote {out_path}")


if __name__ == "__main__":
    main()
