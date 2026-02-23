#!/usr/bin/env python3
"""
Generate compute-accounting summary and LaTeX table.

Input:
  results/paper/compute/compute_runs.csv

Outputs:
  results/paper/tables/compute_summary.csv
  paper/tables/table_compute_accounting.tex
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple


Row = Dict[str, str]


def method_label(method: str) -> str:
    m = method.strip().lower()
    if m == "baseline":
        return "PINN baseline"
    if m == "ours":
        return "RoPINN-ResFF (ours)"
    return method


def task_label(task: str) -> str:
    t = task.strip().lower()
    if t == "reaction":
        return "Reaction"
    if t == "wave":
        return "Wave"
    if t == "convection":
        return "Convection"
    return task


def read_rows(path: str) -> List[Row]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"input csv not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"no rows in input csv: {path}")
    required = {
        "task",
        "method",
        "seed",
        "max_iters",
        "elapsed_sec",
        "sec_per_iter",
        "params",
        "relative_l1",
        "relative_l2",
        "log_path",
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return rows


def to_float(x: str) -> float:
    return float(x)


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.pstdev(vals)


def fmt_num(x: float) -> str:
    return f"{x:.6f}"


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser("Generate compute accounting table")
    parser.add_argument(
        "--input-csv",
        default="results/paper/compute/compute_runs.csv",
        help="raw compute run csv",
    )
    parser.add_argument(
        "--out-csv",
        default="results/paper/tables/compute_summary.csv",
        help="summary csv",
    )
    parser.add_argument(
        "--out-tex",
        default="paper/tables/table_compute_accounting.tex",
        help="latex table path",
    )
    args = parser.parse_args()

    rows = read_rows(args.input_csv)

    grouped: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    for r in rows:
        grouped[(r["task"].strip().lower(), r["method"].strip().lower())].append(r)

    order_tasks = ["reaction", "wave", "convection"]
    order_methods = ["baseline", "ours"]

    summary_rows: List[Dict[str, str]] = []
    for task in order_tasks:
        for method in order_methods:
            key = (task, method)
            if key not in grouped:
                continue
            g = grouped[key]
            elapsed_vals = [to_float(x["elapsed_sec"]) for x in g]
            spi_vals = [to_float(x["sec_per_iter"]) for x in g]
            l1_vals = [to_float(x["relative_l1"]) for x in g if x["relative_l1"] != "NA"]
            l2_vals = [to_float(x["relative_l2"]) for x in g if x["relative_l2"] != "NA"]
            params_vals = [int(x["params"]) for x in g if x["params"] != "NA"]

            elapsed_mean, elapsed_std = mean_std(elapsed_vals)
            spi_mean, spi_std = mean_std(spi_vals)
            l1_mean, l1_std = mean_std(l1_vals) if l1_vals else (float("nan"), float("nan"))
            l2_mean, l2_std = mean_std(l2_vals) if l2_vals else (float("nan"), float("nan"))
            params_mean = int(round(st.mean(params_vals))) if params_vals else -1
            max_iters = int(g[0]["max_iters"])
            n = len(g)

            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "n_runs": str(n),
                    "max_iters": str(max_iters),
                    "params": str(params_mean if params_mean >= 0 else "NA"),
                    "elapsed_mean_sec": fmt_num(elapsed_mean),
                    "elapsed_std_sec": fmt_num(elapsed_std),
                    "sec_per_iter_mean": fmt_num(spi_mean),
                    "sec_per_iter_std": fmt_num(spi_std),
                    "relative_l1_mean": fmt_num(l1_mean) if l1_vals else "NA",
                    "relative_l1_std": fmt_num(l1_std) if l1_vals else "NA",
                    "relative_l2_mean": fmt_num(l2_mean) if l2_vals else "NA",
                    "relative_l2_std": fmt_num(l2_std) if l2_vals else "NA",
                }
            )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task",
                "method",
                "n_runs",
                "max_iters",
                "params",
                "elapsed_mean_sec",
                "elapsed_std_sec",
                "sec_per_iter_mean",
                "sec_per_iter_std",
                "relative_l1_mean",
                "relative_l1_std",
                "relative_l2_mean",
                "relative_l2_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Build LaTeX table
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Compute accounting under equal training budget (1000 iterations). Wall-time is measured end-to-end on the same device. Values are mean $\\pm$ std across repeated runs.}",
        "\\label{tab:compute-accounting}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Task & Method & Params & Wall-time (s) & Sec/iter \\\\",
        "\\midrule",
    ]

    for task in order_tasks:
        task_block = [r for r in summary_rows if r["task"] == task]
        if not task_block:
            continue
        for r in task_block:
            params = r["params"]
            if params != "NA":
                params = f"{int(params):,}"
            elapsed_cell = fmt_pm(float(r["elapsed_mean_sec"]), float(r["elapsed_std_sec"]), digits=2)
            spi_cell = fmt_pm(float(r["sec_per_iter_mean"]), float(r["sec_per_iter_std"]), digits=4)
            lines.append(
                f"{task_label(r['task'])} & {method_label(r['method'])} & {params} & {elapsed_cell} & {spi_cell} \\\\"
            )
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines = lines[:-1]
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    print(f"[OK] Wrote {args.out_csv}")
    print(f"[OK] Wrote {args.out_tex}")


if __name__ == "__main__":
    main()

