#!/usr/bin/env python3
"""
Generate 4 LaTeX tables + summary CSVs for paper experiments.

Default outputs:
  - paper/tables/table_reaction_main.tex
  - paper/tables/table_reaction_mseed.tex
  - paper/tables/table_wave_mseed.tex
  - paper/tables/table_convection_mseed.tex
  - results/paper/tables/seed_rows.csv
  - results/paper/tables/seed_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics as st
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


L1_PATTERN = re.compile(r"relative L1 error:\s*([0-9eE+\-\.]+)")
L2_PATTERN = re.compile(r"relative L2 error:\s*([0-9eE+\-\.]+)")
SEED_PATTERN = re.compile(r"_s(\d+)\.log$")


REACTION_MAIN_RUNS = [
    ("paper_base_reaction_1000", "Original RoPINN PINN baseline (backup)"),
    ("paper_curr_pinn_reaction_1000", "Current PINN (modified branch)"),
    ("paper_ablate_resff_only_1000", "RoPINN-ResFF (ours)"),
    ("paper_best_reaction_1000", "RoPINN-ResFF + curriculum (ours)"),
]


@dataclass
class SeedStats:
    n: int
    l1_mean: float
    l1_std: float
    l2_mean: float
    l2_std: float


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        return fp.read()


def parse_last_metric(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_log(path: str) -> Tuple[Optional[float], Optional[float]]:
    text = read_text(path)
    return parse_last_metric(text, L1_PATTERN), parse_last_metric(text, L2_PATTERN)


def format_float(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or std is None:
        return "NA"
    return f"{mean:.6f} $\\pm$ {std:.6f}"


def maybe_bold(text: str, use_bold: bool) -> str:
    return f"\\textbf{{{text}}}" if use_bold and text != "NA" else text


def load_reaction_main(summary_csv: str, strict: bool) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if not os.path.exists(summary_csv):
        msg = f"summary csv not found: {summary_csv}"
        if strict:
            raise FileNotFoundError(msg)
        warn(msg)
        return out

    with open(summary_csv, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            run = row.get("run", "")
            try:
                l1 = float(row["relative_l1"]) if row.get("relative_l1") else None
                l2 = float(row["relative_l2"]) if row.get("relative_l2") else None
            except ValueError:
                l1, l2 = None, None
            out[run] = (l1, l2)
    return out


def seed_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    match = SEED_PATTERN.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def parse_seed_logs(glob_pattern: str, strict: bool) -> List[Tuple[str, Optional[float], Optional[float]]]:
    paths = sorted(glob.glob(glob_pattern), key=seed_key)
    if not paths:
        msg = f"no files matched: {glob_pattern}"
        if strict:
            raise FileNotFoundError(msg)
        warn(msg)
        return []

    rows: List[Tuple[str, Optional[float], Optional[float]]] = []
    for path in paths:
        l1, l2 = parse_log(path)
        rows.append((path, l1, l2))
    return rows


def calc_seed_stats(rows: Sequence[Tuple[str, Optional[float], Optional[float]]]) -> Optional[SeedStats]:
    l1_vals = [r[1] for r in rows if r[1] is not None]
    l2_vals = [r[2] for r in rows if r[2] is not None]
    if not l1_vals or not l2_vals or len(l1_vals) != len(l2_vals):
        return None
    return SeedStats(
        n=len(l1_vals),
        l1_mean=st.mean(l1_vals),
        l1_std=st.pstdev(l1_vals) if len(l1_vals) > 1 else 0.0,
        l2_mean=st.mean(l2_vals),
        l2_std=st.pstdev(l2_vals) if len(l2_vals) > 1 else 0.0,
    )


def reduction_percent(base: Optional[float], ours: Optional[float]) -> Optional[float]:
    if base is None or ours is None or base == 0:
        return None
    return (base - ours) / base * 100.0


def render_reaction_main_table(
    metrics: Dict[str, Tuple[Optional[float], Optional[float]]],
    output_path: str,
) -> None:
    l1_candidates = [metrics.get(run, (None, None))[0] for run, _ in REACTION_MAIN_RUNS]
    l2_candidates = [metrics.get(run, (None, None))[1] for run, _ in REACTION_MAIN_RUNS]
    best_l1 = min((v for v in l1_candidates if v is not None), default=None)
    best_l2 = min((v for v in l2_candidates if v is not None), default=None)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Main comparison on 1D reaction (1000 iterations, single run).}",
        "\\label{tab:reaction-main}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Relative L1 & Relative L2 \\\\",
        "\\midrule",
    ]

    for run, method_name in REACTION_MAIN_RUNS:
        l1, l2 = metrics.get(run, (None, None))
        l1_txt = format_float(l1)
        l2_txt = format_float(l2)
        l1_txt = maybe_bold(l1_txt, l1 is not None and best_l1 is not None and abs(l1 - best_l1) < 1e-15)
        l2_txt = maybe_bold(l2_txt, l2 is not None and best_l2 is not None and abs(l2 - best_l2) < 1e-15)
        lines.append(f"{method_name} & {l1_txt} & {l2_txt} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def render_two_row_seed_table(
    output_path: str,
    caption: str,
    label: str,
    baseline_name: str,
    ours_name: str,
    baseline_stats: Optional[SeedStats],
    ours_stats: Optional[SeedStats],
) -> None:
    base_l1 = baseline_stats.l1_mean if baseline_stats else None
    ours_l1 = ours_stats.l1_mean if ours_stats else None
    base_l2 = baseline_stats.l2_mean if baseline_stats else None
    ours_l2 = ours_stats.l2_mean if ours_stats else None
    ours_is_better_l1 = (
        base_l1 is not None and ours_l1 is not None and ours_l1 < base_l1
    )
    ours_is_better_l2 = (
        base_l2 is not None and ours_l2 is not None and ours_l2 < base_l2
    )

    base_l1_cell = format_mean_std(
        baseline_stats.l1_mean if baseline_stats else None,
        baseline_stats.l1_std if baseline_stats else None,
    )
    base_l2_cell = format_mean_std(
        baseline_stats.l2_mean if baseline_stats else None,
        baseline_stats.l2_std if baseline_stats else None,
    )
    ours_l1_cell = format_mean_std(
        ours_stats.l1_mean if ours_stats else None,
        ours_stats.l1_std if ours_stats else None,
    )
    ours_l2_cell = format_mean_std(
        ours_stats.l2_mean if ours_stats else None,
        ours_stats.l2_std if ours_stats else None,
    )

    base_l1_cell = maybe_bold(base_l1_cell, not ours_is_better_l1)
    base_l2_cell = maybe_bold(base_l2_cell, not ours_is_better_l2)
    ours_l1_cell = maybe_bold(ours_l1_cell, ours_is_better_l1)
    ours_l2_cell = maybe_bold(ours_l2_cell, ours_is_better_l2)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Relative L1 (mean $\\pm$ std) & Relative L2 (mean $\\pm$ std) \\\\",
        "\\midrule",
        f"{baseline_name} & {base_l1_cell} & {base_l2_cell} \\\\",
        f"{ours_name} & {ours_l1_cell} & {ours_l2_cell} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def seed_count_label(baseline_stats: Optional[SeedStats], ours_stats: Optional[SeedStats]) -> str:
    n_base = baseline_stats.n if baseline_stats else 0
    n_ours = ours_stats.n if ours_stats else 0
    if n_base > 0 and n_ours > 0 and n_base == n_ours:
        return str(n_base)
    if n_base > 0 and n_ours > 0:
        return f"{n_base}/{n_ours}"
    if n_base > 0:
        return str(n_base)
    if n_ours > 0:
        return str(n_ours)
    return "NA"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_seed_rows_csv(
    out_csv: str,
    task: str,
    method: str,
    rows: Sequence[Tuple[str, Optional[float], Optional[float]]],
) -> None:
    mode = "a" if os.path.exists(out_csv) else "w"
    with open(out_csv, mode, newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if mode == "w":
            writer.writerow(["task", "method", "seed", "relative_l1", "relative_l2", "log_path"])
        for path, l1, l2 in rows:
            m = SEED_PATTERN.search(os.path.basename(path))
            seed = int(m.group(1)) if m else ""
            writer.writerow(
                [task, method, seed, l1 if l1 is not None else "", l2 if l2 is not None else "", path]
            )


def write_seed_summary_csv(out_csv: str, rows: List[List[object]]) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "task",
                "method",
                "n",
                "l1_mean",
                "l1_std",
                "l2_mean",
                "l2_std",
                "l1_reduction_vs_baseline_percent",
                "l2_reduction_vs_baseline_percent",
            ]
        )
        writer.writerows(rows)


def append_summary_rows(
    output_rows: List[List[object]],
    task: str,
    baseline_stats: Optional[SeedStats],
    ours_stats: Optional[SeedStats],
) -> None:
    base_l1_red = 0.0
    base_l2_red = 0.0
    ours_l1_red = reduction_percent(
        baseline_stats.l1_mean if baseline_stats else None,
        ours_stats.l1_mean if ours_stats else None,
    )
    ours_l2_red = reduction_percent(
        baseline_stats.l2_mean if baseline_stats else None,
        ours_stats.l2_mean if ours_stats else None,
    )

    output_rows.append(
        [
            task,
            "baseline",
            baseline_stats.n if baseline_stats else "",
            baseline_stats.l1_mean if baseline_stats else "",
            baseline_stats.l1_std if baseline_stats else "",
            baseline_stats.l2_mean if baseline_stats else "",
            baseline_stats.l2_std if baseline_stats else "",
            base_l1_red,
            base_l2_red,
        ]
    )
    output_rows.append(
        [
            task,
            "ours",
            ours_stats.n if ours_stats else "",
            ours_stats.l1_mean if ours_stats else "",
            ours_stats.l1_std if ours_stats else "",
            ours_stats.l2_mean if ours_stats else "",
            ours_stats.l2_std if ours_stats else "",
            ours_l1_red if ours_l1_red is not None else "",
            ours_l2_red if ours_l2_red is not None else "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper LaTeX tables from logs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--summary-csv", default="results/paper/summary.csv")
    parser.add_argument("--strict", action="store_true", help="Fail if required files are missing")
    args = parser.parse_args()

    tables_dir = os.path.join(args.paper_dir, "tables")
    stats_dir = os.path.join(args.results_dir, "paper", "tables")
    ensure_dir(tables_dir)
    ensure_dir(stats_dir)

    # 1) Reaction main single-run table.
    reaction_main_metrics = load_reaction_main(args.summary_csv, strict=args.strict)
    render_reaction_main_table(
        reaction_main_metrics,
        os.path.join(tables_dir, "table_reaction_main.tex"),
    )

    # 2) Multi-seed blocks.
    patterns = {
        "reaction": {
            "baseline": os.path.join(args.results_dir, "mseed_react_base_s*.log"),
            "ours": os.path.join(args.results_dir, "mseed_react_resff_s*.log"),
            "caption_tpl": "Reaction benchmark with {n} seeds (1000 iterations).",
            "label": "tab:reaction-mseed",
            "outfile": os.path.join(tables_dir, "table_reaction_mseed.tex"),
        },
        "wave": {
            "baseline": os.path.join(args.results_dir, "mseed_wave_base_s*.log"),
            "ours": os.path.join(args.results_dir, "mseed_wave_resff_s*.log"),
            "caption_tpl": "Wave benchmark with {n} seeds (1000 iterations).",
            "label": "tab:wave-mseed",
            "outfile": os.path.join(tables_dir, "table_wave_mseed.tex"),
        },
        "convection": {
            "baseline": os.path.join(args.results_dir, "mseed_conv_base_s*.log"),
            "ours": os.path.join(args.results_dir, "mseed_conv_resff_s*.log"),
            "caption_tpl": "Convection benchmark with {n} seeds (1000 iterations).",
            "label": "tab:conv-mseed",
            "outfile": os.path.join(tables_dir, "table_convection_mseed.tex"),
        },
    }

    seed_rows_csv = os.path.join(stats_dir, "seed_rows.csv")
    if os.path.exists(seed_rows_csv):
        os.remove(seed_rows_csv)

    summary_rows: List[List[object]] = []
    for task, cfg in patterns.items():
        base_rows = parse_seed_logs(cfg["baseline"], strict=args.strict)
        ours_rows = parse_seed_logs(cfg["ours"], strict=args.strict)
        write_seed_rows_csv(seed_rows_csv, task, "baseline", base_rows)
        write_seed_rows_csv(seed_rows_csv, task, "ours", ours_rows)

        base_stats = calc_seed_stats(base_rows)
        ours_stats = calc_seed_stats(ours_rows)
        caption = cfg["caption_tpl"].format(n=seed_count_label(base_stats, ours_stats))
        render_two_row_seed_table(
            output_path=cfg["outfile"],
            caption=caption,
            label=cfg["label"],
            baseline_name="Baseline PINN",
            ours_name="RoPINN-ResFF (ours)",
            baseline_stats=base_stats,
            ours_stats=ours_stats,
        )
        append_summary_rows(summary_rows, task, base_stats, ours_stats)

    write_seed_summary_csv(os.path.join(stats_dir, "seed_summary.csv"), summary_rows)

    print("[OK] Generated LaTeX tables:")
    print(f"  - {os.path.join(tables_dir, 'table_reaction_main.tex')}")
    print(f"  - {os.path.join(tables_dir, 'table_reaction_mseed.tex')}")
    print(f"  - {os.path.join(tables_dir, 'table_wave_mseed.tex')}")
    print(f"  - {os.path.join(tables_dir, 'table_convection_mseed.tex')}")
    print("[OK] Generated stats CSVs:")
    print(f"  - {seed_rows_csv}")
    print(f"  - {os.path.join(stats_dir, 'seed_summary.csv')}")


if __name__ == "__main__":
    main()
