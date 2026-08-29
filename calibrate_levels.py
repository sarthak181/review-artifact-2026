"""
Measure the nominal-bbox-fraction -> true-object-coverage mapping from the
segmentation polygons, and invert it to pick occlusion levels that are evenly
spaced on the axis that actually means something.

WHY
  The first graded render used nominal levels 25/50/75/100 (% of bounding box).
  Measured against the polygons, those land at 43/76/97/100 % of the object's
  PIXELS: almost no resolution above 76%, and the whole 0-43% region unsampled.
  The mapping is strongly non-linear because objects concentrate their mass near
  the centre of their box, so a centred box is disproportionately effective.

  This script sweeps nominal fractions finely, measures true coverage at each,
  and reports the nominal values needed to hit evenly spaced true targets.

  python calibrate_levels.py
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from build_graded import dilate, safe_id, scaled_box

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = [2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 36, 42, 50, 58, 68, 80, 90, 100]
TARGETS = [0.20, 0.40, 0.60, 0.80, 0.95, 1.00]


def raster(shapes, W, H, polygons=True):
    m = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(m)
    for s in shapes:
        if polygons:
            for poly in s:
                if len(poly) >= 6:
                    d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)],
                              fill=1)
        else:
            d.rectangle(s, fill=1)
    return np.array(m, dtype=bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=os.path.join(HERE, "dataset", "probe_graded"))
    p.add_argument("--match", default=os.path.join(HERE, "results", "coco_match.json"))
    p.add_argument("--coco-ann", default=os.path.join(HERE, "dataset", "coco",
                                                      "instances_val2014.json"))
    p.add_argument("--sample", type=int, default=150)
    args = p.parse_args()

    man = json.load(open(os.path.join(args.probe, "manifest.json"), encoding="utf-8"))[: args.sample]
    matched = json.load(open(args.match, encoding="utf-8"))
    by_sid = {safe_id(k): v for k, v in matched.items()}
    coco = json.load(open(args.coco_ann, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    images = {im["id"]: im for im in coco["images"]}

    rows = {g: [] for g in GRID}
    used = 0
    for m in man:
        meta = by_sid.get(m["item_id"])
        if meta is None:
            continue
        im = images[meta["coco_image_id"]]
        W, H = im["width"], im["height"]
        anns = [ann_by_id[i] for i in meta["target_ann_ids"] if i in ann_by_id]
        anns = [a for a in anns
                if isinstance(a.get("segmentation"), list) and a["segmentation"]]
        if not anns:
            continue
        obj = raster([a["segmentation"] for a in anns], W, H)
        n_obj = int(obj.sum())
        if n_obj == 0:
            continue
        full = [dilate(a["bbox"], W, H) for a in anns]
        for g in GRID:
            boxes = [scaled_box(b, g / 100.0, W, H) for b in full]
            bm = raster(boxes, W, H, polygons=False)
            rows[g].append(float((obj & bm).sum()) / n_obj)
        used += 1

    print(f"calibrated on {used} items\n")
    print(f"{'nominal':>8}  {'true coverage':>26}")
    print(f"{'(%bbox)':>8}  {'median':>8} {'p25':>8} {'p75':>8}")
    print("-" * 38)
    med = {}
    for g in GRID:
        v = np.array(rows[g])
        med[g] = float(np.median(v))
        print(f"{g:7}%  {med[g]:8.3f} {np.percentile(v,25):8.3f} {np.percentile(v,75):8.3f}")

    print("\nINVERTED: nominal needed for evenly spaced TRUE coverage:")
    xs = sorted(med)
    ys = [med[g] for g in xs]
    chosen = []
    for t in TARGETS:
        pick = None
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
            if y0 <= t <= y1 and y1 > y0:
                pick = x0 + (t - y0) * (x1 - x0) / (y1 - y0)
                break
        if pick is None:
            pick = xs[-1] if t >= ys[-1] else xs[0]
        chosen.append(int(round(pick)))
        print(f"  true {t:.2f}  ->  nominal {round(pick):3d}% of bbox")

    uniq = sorted(set(chosen))
    print(f"\nRECOMMENDED LEVELS = {tuple(uniq)}")
    print("  (existing run used (25, 50, 75, 100) -> true 43/76/97/100)")


if __name__ == "__main__":
    main()
