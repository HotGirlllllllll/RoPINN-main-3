#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


L1_PATTERN = re.compile(r"relative L1 error:\s*([0-9eE+\-\.]+)")
L2_PATTERN = re.compile(r"relative L2 error:\s*([0-9eE+\-\.]+)")


def parse_metric(log_text: str, pattern: re.Pattern) -> Optional[float]:
    matches = pattern.findall(log_text)
    if not matches:
        return None
    return float(matches[-1])


def parse_log(path: str) -> Tuple[Optional[float], Optional[float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        text = fp.read()
    return parse_metric(text, L1_PATTERN), parse_metric(text, L2_PATTERN)


def to_table(rows: List[dict]) -> str:
    lines = [
        "| run | relative_l1 | relative_l2 |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        l1 = f"{row['relative_l1']:.6f}" if row["relative_l1"] is not None else "NA"
        l2 = f"{row['relative_l2']:.6f}" if row["relative_l2"] is not None else "NA"
        lines.append(f"| {row['run']} | {l1} | {l2} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser("Collect paper metrics from logs")
    parser.add_argument("--glob", dest="glob_pattern", default="results/paper_*.log")
    parser.add_argument("--outdir", default="results/paper")
    parser.add_argument("--baseline", default="")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob_pattern))
    if not paths:
        raise SystemExit(f"No log files matched: {args.glob_pattern}")

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for path in paths:
        l1, l2 = parse_log(path)
        rows.append(
            {
                "run": os.path.splitext(os.path.basename(path))[0],
                "relative_l1": l1,
                "relative_l2": l2,
                "log_path": path,
            }
        )

    rows.sort(key=lambda item: float("inf") if item["relative_l2"] is None else item["relative_l2"])

    csv_path = os.path.join(args.outdir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["run", "relative_l1", "relative_l2", "log_path"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(args.outdir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(to_table(rows))

    valid_rows = [row for row in rows if row["relative_l1"] is not None and row["relative_l2"] is not None]
    if valid_rows and plt is not None:
        labels = [row["run"] for row in valid_rows]
        l1_vals = [row["relative_l1"] for row in valid_rows]
        l2_vals = [row["relative_l2"] for row in valid_rows]
        x = list(range(len(labels)))

        fig = plt.figure(figsize=(max(6, len(labels) * 1.5), 4))
        ax = fig.add_subplot(111)
        width = 0.38
        ax.bar([i - width / 2 for i in x], l1_vals, width=width, label="L1", color="tab:blue")
        ax.bar([i + width / 2 for i in x], l2_vals, width=width, label="L2", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Relative Error")
        ax.set_title("Main Comparison on 1D Reaction")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "summary_bar.pdf"), bbox_inches="tight")

    if args.baseline:
        base = next((row for row in rows if row["run"] == args.baseline), None)
        if base and base["relative_l1"] is not None and base["relative_l2"] is not None:
            red_path = os.path.join(args.outdir, "reduction_vs_baseline.csv")
            with open(red_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["run", "l1_reduction_percent", "l2_reduction_percent"])
                for row in rows:
                    if row["relative_l1"] is None or row["relative_l2"] is None:
                        continue
                    l1_red = (base["relative_l1"] - row["relative_l1"]) / base["relative_l1"] * 100.0
                    l2_red = (base["relative_l2"] - row["relative_l2"]) / base["relative_l2"] * 100.0
                    writer.writerow([row["run"], f"{l1_red:.4f}", f"{l2_red:.4f}"])


if __name__ == "__main__":
    main()
