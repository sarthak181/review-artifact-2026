"""Run the graded-occlusion probe: 6 conditions per question, per quant level."""
import argparse
import os

from unkvqa.probe import GRADED_CONDITIONS, image_path, load_graded
from unkvqa.protocols import DEFAULT_VARIANT, PROMPT_VARIANTS
from unkvqa.runner import run

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-root", default=os.path.join(HERE, "dataset", "probe_graded"))
    p.add_argument("--model-path", default=os.path.join(HERE, "models", "qwen25vl-3b"))
    p.add_argument("--quant", default="bf16")
    p.add_argument("--protocols", default="freeform")
    p.add_argument("--out", default=os.path.join(HERE, "results", "graded.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(PROMPT_VARIANTS))
    p.add_argument("--conditions", default="",
                   help="comma list; default = all 8. Use "
                        "'original,target_100,control' for refusal-content work, where the "
                        "partial-occlusion levels contribute nothing and free-form generation "
                        "is ~10x slower than emitting a canonical token")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    quants = [q.strip() for q in args.quant.split(",") if q.strip()]
    protos = [x.strip() for x in args.protocols.split(",") if x.strip()]
    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip()) or GRADED_CONDITIONS
    unknown = [c for c in conds if c not in GRADED_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; valid: {list(GRADED_CONDITIONS)}")
    items = load_graded(args.probe_root, limit=args.limit, seed=args.seed)
    n = len(items) * len(conds) * len(protos) * len(quants)
    print(f"graded items: {len(items)}  conditions: {list(conds)}")
    print(f"plan: {n} records -> {args.out}\n")

    run(items=items, dataset_root=args.probe_root, model_path=args.model_path,
        quant_levels=quants, protos=protos, out_path=args.out,
        batch_size=args.batch_size or None, resume=not args.no_resume,
        variant=args.variant, conditions=conds, path_fn=image_path)


if __name__ == "__main__":
    main()
