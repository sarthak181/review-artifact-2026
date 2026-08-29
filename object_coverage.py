"""
Convert the dose-response x-axis from "percent of a bounding box" into
"percent of the object's actual pixels".

WHY
  The graded probe scales a rectangle to cover 25/50/75/100% of the target's
  BOUNDING BOX. A bounding box is mostly background for anything that is not
  rectangular and axis-aligned: a carrot lying diagonally across a plate has a
  box that is largely plate. So "25% occluded" can mean anything from 10% to 40%
  of the object actually gone, depending on shape and orientation.

  That matters because the 50%-abstention threshold is the headline number, and
  right now its units are uninterpretable. Rescaling the axis to true object
  coverage makes the threshold mean something a reader can hold: "this model
  needs roughly N% of an object to disappear before it stops answering."

  It also corrects a bias the pictures made visible: because masks grow from the
  object's CENTRE, and identity often lives at the extremities (a carrot's ends,
  a bottle's label), low nominal levels are weaker interventions than their
  percentage suggests.

METHOD
  Rasterise the COCO segmentation polygons for the target category, apply the
  exact same box geometry build_graded.py used, and measure the intersection.
  Purely post-hoc over rendered data: no GPU, no re-rendering.

  python object_coverage.py --probe dataset/probe_graded
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

from build_graded import LEVELS, dilate, safe_id, scaled_box

HERE = os.path.dirname(os.path.abspath(__file__))


def rasterize(anns, W, H):
    """Union of every polygon for the target category, as a binary mask."""
    m = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return m


def box_mask(boxes, W, H):
    m = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(m)
    for b in boxes:
        d.rectangle(b, fill=1)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=os.path.join(HERE, "dataset", "probe_graded"))
    p.add_argument("--match", default=os.path.join(HERE, "results", "coco_match.json"))
    p.add_argument("--coco-ann", default=os.path.join(HERE, "dataset", "coco",
                                                      "instances_val2014.json"))
    p.add_argument("--out", default="")
    args = p.parse_args()
    out_path = args.out or os.path.join(args.probe, "object_coverage.json")

    man = json.load(open(os.path.join(args.probe, "manifest.json"), encoding="utf-8"))
    matched = json.load(open(args.match, encoding="utf-8"))
    by_sid = {safe_id(k): v for k, v in matched.items()}
    coco = json.load(open(args.coco_ann, encoding="utf-8"))
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    images = {im["id"]: im for im in coco["images"]}

    cov, skipped = {}, 0
    for m in man:
        meta = by_sid.get(m["item_id"])
        if meta is None:
            skipped += 1
            continue
        W, H = images[meta["coco_image_id"]]["width"], images[meta["coco_image_id"]]["height"]
        anns = [ann_by_id[i] for i in meta["target_ann_ids"] if i in ann_by_id]
        anns = [a for a in anns if isinstance(a.get("segmentation"), list) and a["segmentation"]]
        if not anns:
            skipped += 1
            continue

        obj = rasterize(anns, W, H)
        obj_px = sum(obj.getdata())
        if obj_px == 0:
            skipped += 1
            continue
        full = [dilate(a["bbox"], W, H) for a in anns]

        entry = {"object_px": obj_px, "image_px": W * H,
                 "bbox_fill": None, "levels": {}}
        # How much of the box is actually object? 1.0 = perfectly tight box.
        bm = box_mask(full, W, H)
        inter_full = sum(1 for o, b in zip(obj.getdata(), bm.getdata()) if o and b)
        box_px = sum(bm.getdata())
        entry["bbox_fill"] = round(inter_full / box_px, 4) if box_px else None

        for lvl in LEVELS:
            boxes = [scaled_box(b, lvl / 100.0, W, H) for b in full]
            bm = box_mask(boxes, W, H)
            inter = sum(1 for o, b in zip(obj.getdata(), bm.getdata()) if o and b)
            entry["levels"][str(lvl)] = round(inter / obj_px, 4)
        cov[m["item_id"]] = entry

    json.dump(cov, open(out_path, "w", encoding="utf-8"), indent=1)
    print(f"computed object coverage for {len(cov)} items ({skipped} skipped)")
    print(f"wrote {out_path}\n")

    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2] if v else float("nan")

    print(f"{'nominal':>9}  {'true object coverage':>34}")
    print(f"{'(of bbox)':>9}  {'median':>8} {'p25':>8} {'p75':>8}")
    print("-" * 42)
    for lvl in LEVELS:
        vals = sorted(e["levels"][str(lvl)] for e in cov.values())
        n = len(vals)
        print(f"{lvl:8}%  {med(vals):8.3f} {vals[n//4]:8.3f} {vals[3*n//4]:8.3f}")
    fills = [e["bbox_fill"] for e in cov.values() if e["bbox_fill"] is not None]
    print(f"\nbbox tightness (object px / box px): median={med(fills):.3f}")
    print("  1.0 would mean boxes contain only the object; lower means the")
    print("  nominal axis overstates how much of the object is removed.")


if __name__ == "__main__":
    main()
