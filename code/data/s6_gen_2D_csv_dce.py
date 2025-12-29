import csv
import argparse
from pathlib import Path

SPLIT_TOKENS = ("train", "val", "valid", "validation", "test")


def infer_split(parts):
    parts_lower = [p.lower() for p in parts]
    for p in parts_lower:
        if p in ("val", "valid", "validation"):
            return "val"
        if p in ("train", "test"):
            return p
    # also catch tokens embedded in names like "trainA"
    for p in parts_lower:
        for tok in SPLIT_TOKENS:
            if tok in p:
                if tok in ("val", "valid", "validation"):
                    return "val"
                return "train" if tok == "train" else "test"
    return ""


def make_csv(root_dir, out_csv,
             dce1_name="_dce1.jpg",
             dce2_name="_dce2.jpg",
             dce3_name="_dce3.jpg"):
    root = Path(root_dir).resolve()
    rows = []

    

    for slice_dir in root.rglob("*"):
        if not slice_dir.is_dir():
            continue

        # Look for triplet files
        dce1 = list(slice_dir.glob(f"*{dce1_name}"))
        dce2 = list(slice_dir.glob(f"*{dce2_name}"))
        dce3 = list(slice_dir.glob(f"*{dce3_name}"))

        if dce1 and dce2 and dce3:
            dce1_path, dce2_path, dce3_path = dce1[0], dce2[0], dce3[0]

            exam_dir = slice_dir.parent
            exam_id = exam_dir.name
            slice_id = slice_dir.name

            # components relative to root to infer dataset/subject/split
            rel_parts = dce1_path.relative_to(root).parts
            dataset = rel_parts[0] if len(rel_parts) >= 1 else ""
            subject = rel_parts[1] if len(rel_parts) >= 2 else ""
            split = infer_split(rel_parts)

            rows.append([
                dataset,
                subject,
                exam_id,
                slice_id,
                split,
                str(dce1_path),
                str(dce2_path),
                str(dce3_path),
            ])

    # sort for stability
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    # write CSV
    header = ["Dataset", "Subject", "ExamID", "Slice", "Split", "DCE1", "DCE2", "DCE3"]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Found {len(rows)} triplets. CSV saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index DCE triplets (dce1/dce2/dce3) into a CSV (one row per slice)."
    )
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--out", type=str, default="dataset_triplets.csv", help="Output CSV file path")
    parser.add_argument("--dce1-name", type=str, default="_dce1.jpg", help="Filename suffix for DCE1")
    parser.add_argument("--dce2-name", type=str, default="_dce2.jpg", help="Filename suffix for DCE2")
    parser.add_argument("--dce3-name", type=str, default="_dce3.jpg", help="Filename suffix for DCE3")
    args = parser.parse_args()

    make_csv(args.root, args.out, args.dce1_name, args.dce2_name, args.dce3_name)
