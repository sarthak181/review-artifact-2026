"""
Step 2: render the constructed probe set: three image variants per question.

  original          no mask                  -> should ANSWER
  target-masked     answer object removed    -> should ABSTAIN
  control-masked    a DIFFERENT object of matched area removed -> should ANSWER

DESIGN MOTIVATION
  The control carries a mask of comparable size to the target intervention. This
  reduces sensitivity to the visible masking artifact while varying whether the
  annotated answer object survives. The complete-removal label remains an
  operational assumption: contextual evidence and annotation omissions can make
  individual items ambiguous, so human validation is a priority for future work.

  It is also a WITHIN-ITEM design: every question yields both a target and a
  should-answer control measurement on the same image and wording, reducing
  between-item variance in the target/control contrast.

TWO RENDERING DECISIONS THAT MATTER
  1. BOUNDING BOX, NOT SILHOUETTE. Masking the exact polygon leaves a
     dog-shaped hole, and the silhouette alone can identify the object: the
     probe would leak the answer it is trying to remove. Boxes destroy shape.
  2. AREA MATCHED ON WHAT IS ACTUALLY PAINTED. Because boxes are masked, the
     control is chosen and reported by painted box area. The match is approximate;
     the release analysis includes stricter area-ratio sensitivity subsets.

  python build_probe.py
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
COCO_IMG_URL = "http://images.cocodataset.org/val2014/{name}"


def safe_id(key: str) -> str:
    return key.replace(":", "_").replace(".jpg", "")


def union_box(boxes):
    xs0 = min(b[0] for b in boxes); ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes); ys1 = max(b[1] + b[3] for b in boxes)
    return [xs0, ys0, xs1 - xs0, ys1 - ys0]


def dilate(box, w, h, pad_frac=0.04):
    x, y, bw, bh = box
    px, py = bw * pad_frac, bh * pad_frac
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    return [x0, y0, x1, y1]


def paint(img: Image.Image, boxes) -> tuple[Image.Image, float]:
    """Fill boxes with black; return the image and the fraction of pixels covered."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    covered = Image.new("1", out.size, 0)
    cd = ImageDraw.Draw(covered)
    for b in boxes:
        d.rectangle(b, fill=(0, 0, 0))
        cd.rectangle(b, fill=1)
    frac = sum(covered.getdata()) / (out.width * out.height)
    return out, frac


def ensure_image(session, name, path, retries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "cached"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(COCO_IMG_URL.format(name=name), timeout=120)
            if r.status_code != 200:
                raise IOError(f"HTTP {r.status_code}")
            if not r.content.startswith(b"\xff\xd8\xff"):
                raise IOError("not a JPEG")
            tmp = path + ".part"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            os.replace(tmp, path)
            return "ok"
        except Exception:
            if attempt == retries:
                return "fail"
            time.sleep(2 * attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default=os.path.join(HERE, "results", "coco_match.json"))
    p.add_argument("--coco-ann", default=os.path.join(HERE, "dataset", "coco",
                                                      "instances_val2014.json"))
    p.add_argument("--src-images", default=os.path.join(HERE, "dataset", "coco", "val2014"))
    p.add_argument("--out", default=os.path.join(HERE, "dataset", "probe"))
    p.add_argument("--max-area", type=float, default=0.35,
                   help="reject an item if either variant paints more than this "
                        "fraction of the image: the failure mode we are escaping")
    p.add_argument("--max-area-ratio", type=float, default=2.5,
                   help="reject if target/control painted areas differ by more "
                        "than this factor; the artifact must be comparable")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    matched = json.load(open(args.match, encoding="utf-8"))
    if args.limit:
        matched = dict(list(matched.items())[: args.limit])
    print(f"[probe] {len(matched)} candidate items")

    coco = json.load(open(args.coco_ann, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}

    img_dir = os.path.join(args.out, "images")
    os.makedirs(args.src_images, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # 1. fetch the originals we lack
    need = sorted({m["image_file"] for m in matched.values()})
    print(f"[probe] ensuring {len(need)} source images...")
    stats = {"ok": 0, "cached": 0, "fail": 0}
    with requests.Session() as s, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(ensure_image, s, n, os.path.join(args.src_images, n)): n
                for n in need}
        for i, f in enumerate(as_completed(futs), 1):
            stats[f.result()] += 1
            if i % 300 == 0 or i == len(need):
                print(f"   {i}/{len(need)}  ok={stats['ok']} cached={stats['cached']} "
                      f"fail={stats['fail']}")
    if stats["fail"]:
        print(f"   WARNING {stats['fail']} source images failed to download")

    # 2. render the three variants
    manifest, rejected = [], {"too_big": 0, "unbalanced": 0, "missing_src": 0, "bad_ann": 0}
    for key, m in matched.items():
        sid = safe_id(key)
        src = os.path.join(args.src_images, m["image_file"])
        if not os.path.exists(src):
            rejected["missing_src"] += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            rejected["missing_src"] += 1
            continue
        W, H = img.size

        t_anns = [ann_by_id[i] for i in m["target_ann_ids"] if i in ann_by_id]
        c_ann = ann_by_id.get(m["control_ann_id"])
        if not t_anns or c_ann is None:
            rejected["bad_ann"] += 1
            continue

        t_boxes = [dilate(a["bbox"], W, H) for a in t_anns]
        c_boxes = [dilate(c_ann["bbox"], W, H)]

        t_img, t_frac = paint(img, t_boxes)
        c_img, c_frac = paint(img, c_boxes)

        if max(t_frac, c_frac) > args.max_area:
            rejected["too_big"] += 1
            continue
        lo, hi = sorted((t_frac, c_frac))
        if lo <= 0 or hi / lo > args.max_area_ratio:
            rejected["unbalanced"] += 1
            continue

        orig_name, t_name, c_name = f"{sid}_orig.jpg", f"{sid}_target.jpg", f"{sid}_control.jpg"
        img.save(os.path.join(img_dir, orig_name), quality=95)
        t_img.save(os.path.join(img_dir, t_name), quality=95)
        c_img.save(os.path.join(img_dir, c_name), quality=95)

        manifest.append({
            "item_id": sid, "split": m.get("split", "val"),
            "question": m["question"], "answer": m["answer"],
            "image_original": orig_name,
            "image_target": t_name,      # answer object removed -> should ABSTAIN
            "image_control": c_name,     # other object removed  -> should ANSWER
            "target_category": m["target_category"],
            "control_category": m["control_category"],
            "target_masked_frac": round(t_frac, 4),
            "control_masked_frac": round(c_frac, 4),
            "source_image": m["image_file"],
        })

    out_manifest = os.path.join(args.out, "manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\n[probe] rendered {len(manifest)} items -> {img_dir}")
    print(f"[probe] rejected: {rejected}")
    if manifest:
        tf = sorted(x["target_masked_frac"] for x in manifest)
        cf = sorted(x["control_masked_frac"] for x in manifest)
        print(f"  target  painted frac: median={tf[len(tf)//2]:.3f} max={tf[-1]:.3f}")
        print(f"  control painted frac: median={cf[len(cf)//2]:.3f} max={cf[-1]:.3f}")
        print(f"  (UNK-VQA I-2 for comparison: median 0.625)")
    print(f"[probe] wrote {out_manifest}")


if __name__ == "__main__":
    main()
