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

def make_csv(root_dir, out_csv, ct_name="CT.jpg", ctc_name="CTC.jpg"):
    root = Path(root_dir).resolve()
    rows = []

    # find every directory that contains BOTH CT.jpg and CTC.jpg
    for slice_dir in root.rglob("*"):
        if not slice_dir.is_dir():
            continue
        ct_path = slice_dir / ct_name
        ctc_path = slice_dir / ctc_name
        if ct_path.is_file() and ctc_path.is_file():
            exam_dir = slice_dir.parent
            exam_id = exam_dir.name
            slice_id = slice_dir.name

            # components relative to root to infer dataset/subject/split
            rel_parts = ct_path.relative_to(root).parts
            # expect something like: Dataset / Subject / ... / ExamID / Slice / CT.jpg
            dataset = rel_parts[0] if len(rel_parts) >= 1 else ""
            subject = rel_parts[1] if len(rel_parts) >= 2 else ""

            split = infer_split(rel_parts)

            rows.append([
                dataset,
                subject,
                exam_id,
                slice_id,
                split,
                str(ct_path),
                str(ctc_path),
            ])

    # sort for stability: by Dataset, Subject, ExamID, Slice
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    # write CSV
    header = ["Dataset", "Subject", "ExamID", "Slice", "Splits", "CT", "CTC"]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Found {len(rows)} pairs. CSV saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index paired CT/CTC jpgs into a CSV (one row per slice pair).")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--out", type=str, default="dataset_pairs.csv", help="Output CSV file path")
    parser.add_argument("--ct-name", type=str, default="CT.jpg", help="Filename for the CT image")
    parser.add_argument("--ctc-name", type=str, default="CTC.jpg", help="Filename for the CTC image")
    args = parser.parse_args()

    make_csv(args.root, args.out, args.ct_name, args.ctc_name)
