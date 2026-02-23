#!/usr/bin/env python3
"""
Compute statistical tests and confidence intervals for baseline vs ours.

Input:
  results/paper/tables/seed_rows.csv

Output:
  results/paper/tables/stat_tests.csv
  paper/tables/table_stat_tests.tex
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MetricStats:
    task: str
    metric: str
    n: int
    base_mean: float
    base_std: float
    ours_mean: float
    ours_std: float
    delta_mean: float
    ci_low: float
    ci_high: float
    p_two_sided: float


def load_seed_rows(path: str) -> Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]:
    """
    Returns:
      data[task][method][seed] = (l1, l2)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"seed rows csv not found: {path}")

    data: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        required = {"task", "method", "seed", "relative_l1", "relative_l2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            task = row["task"].strip()
            method = row["method"].strip().lower()
            seed = int(row["seed"])
            l1 = float(row["relative_l1"])
            l2 = float(row["relative_l2"])
            data[task][method][seed] = (l1, l2)
    return data


def paired_randomization_test_two_sided(diffs: List[float]) -> float:
    """
    Exact two-sided paired randomization test via sign flips.
    """
    n = len(diffs)
    obs = abs(st.mean(diffs))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        m = abs(st.mean([s * d for s, d in zip(signs, diffs)]))
        if m >= obs - 1e-15:
            extreme += 1
        total += 1
    return extreme / total


def paired_bootstrap_ci(
    diffs: List[float], n_boot: int = 20000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean paired difference.
    """
    n = len(diffs)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        means.append(st.mean(diffs[i] for i in idxs))
    means.sort()
    lo_idx = max(0, int((alpha / 2.0) * n_boot) - 1)
    hi_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot) - 1)
    return means[lo_idx], means[hi_idx]


def summarize_task(
    task: str,
    base_by_seed: Dict[int, Tuple[float, float]],
    ours_by_seed: Dict[int, Tuple[float, float]],
    n_boot: int,
    bootstrap_seed: int,
) -> List[MetricStats]:
    common = sorted(set(base_by_seed) & set(ours_by_seed))
    if not common:
        raise ValueError(f"no common seeds for task={task}")

    base_l1 = [base_by_seed[s][0] for s in common]
    base_l2 = [base_by_seed[s][1] for s in common]
    ours_l1 = [ours_by_seed[s][0] for s in common]
    ours_l2 = [ours_by_seed[s][1] for s in common]

    def pack(metric: str, bvals: List[float], ovals: List[float], seed_shift: int) -> MetricStats:
        diffs = [o - b for b, o in zip(bvals, ovals)]  # ours - baseline
        ci_low, ci_high = paired_bootstrap_ci(
            diffs, n_boot=n_boot, alpha=0.05, seed=bootstrap_seed + seed_shift
        )
        return MetricStats(
            task=task,
            metric=metric,
            n=len(diffs),
            base_mean=st.mean(bvals),
            base_std=st.pstdev(bvals) if len(bvals) > 1 else 0.0,
            ours_mean=st.mean(ovals),
            ours_std=st.pstdev(ovals) if len(ovals) > 1 else 0.0,
            delta_mean=st.mean(diffs),
            ci_low=ci_low,
            ci_high=ci_high,
            p_two_sided=paired_randomization_test_two_sided(diffs),
        )

    return [pack("L1", base_l1, ours_l1, 17), pack("L2", base_l2, ours_l2, 31)]


def fmt_num(x: float) -> str:
    return f"{x:.6f}"


def fmt_p(x: float) -> str:
    if x < 1e-3:
        return "<0.001"
    return f"{x:.4f}"


def write_csv(path: str, rows: List[MetricStats]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "task",
                "metric",
                "n",
                "baseline_mean",
                "baseline_std",
                "ours_mean",
                "ours_std",
                "delta_mean_ours_minus_baseline",
                "ci95_low",
                "ci95_high",
                "p_value_two_sided_paired_randomization",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.task,
                    r.metric,
                    r.n,
                    r.base_mean,
                    r.base_std,
                    r.ours_mean,
                    r.ours_std,
                    r.delta_mean,
                    r.ci_low,
                    r.ci_high,
                    r.p_two_sided,
                ]
            )


def write_latex(path: str, rows: List[MetricStats]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Statistical test summary across paired multi-seed runs. $\\Delta=\\mathrm{mean}(\\mathrm{ours}-\\mathrm{baseline})$; negative values indicate improvement. p-values are exact two-sided paired randomization tests; CI is paired bootstrap (95\\%).}",
        "\\label{tab:stat-tests}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Task & Metric & $\\Delta$ (ours-baseline) & 95\\% CI of $\\Delta$ & p-value \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.task.capitalize()} & {r.metric} & {fmt_num(r.delta_mean)} & [{fmt_num(r.ci_low)}, {fmt_num(r.ci_high)}] & {fmt_p(r.p_two_sided)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser("Generate statistical tests and CI table")
    parser.add_argument(
        "--seed-rows",
        default="results/paper/tables/seed_rows.csv",
        help="CSV with per-seed rows",
    )
    parser.add_argument(
        "--out-csv",
        default="results/paper/tables/stat_tests.csv",
        help="Output CSV for stats",
    )
    parser.add_argument(
        "--out-tex",
        default="paper/tables/table_stat_tests.tex",
        help="Output LaTeX table",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    data = load_seed_rows(args.seed_rows)
    out_rows: List[MetricStats] = []
    for task in ("reaction", "wave", "convection"):
        if task not in data:
            continue
        if "baseline" not in data[task] or "ours" not in data[task]:
            continue
        out_rows.extend(
            summarize_task(
                task=task,
                base_by_seed=data[task]["baseline"],
                ours_by_seed=data[task]["ours"],
                n_boot=args.bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            )
        )

    if not out_rows:
        raise RuntimeError("No valid task rows found for statistical testing.")

    write_csv(args.out_csv, out_rows)
    write_latex(args.out_tex, out_rows)
    print(f"[OK] Wrote {args.out_csv}")
    print(f"[OK] Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
