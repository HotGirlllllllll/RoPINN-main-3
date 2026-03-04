#!/usr/bin/env python3
"""
Build Reaction+Wave Table-1 style comparison artifacts for RoPINN-ResFF.

Inputs:
1) Paper baseline constants (CSV).
2) Metrics CSVs produced by current runs with --paper_outputs.

Outputs:
1) Summary CSV (model-wise comparison and promotion).
2) Win/Loss CSV (task-wise pass/fail under strict dual-metric criterion).
3) LaTeX table for paper inclusion.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


OBJECTIVE_ORDER: Tuple[str, ...] = ("Vanilla", "gPINN", "vPINN", "RoPINN")
TASKS_ALL: Tuple[str, ...] = ("reaction", "wave")
MODEL_TO_OURS = {
    "PINN": "PINN_ResFF",
    "QRes": "QRes_FF",
    "FLS": "FLS_FF",
    "PINNsFormer": "PINNsFormer_FF",
    "KAN": "KAN_FF",
}
MODEL_DISPLAY = {
    "PINN": "PINN",
    "QRes": "QRes",
    "FLS": "FLS",
    "PINNsFormer": "PINNs-Former",
    "KAN": "KAN",
}
TASK_COLS = {
    "reaction": ("reaction_loss", "reaction_rmae", "reaction_rmse"),
    "wave": ("wave_loss", "wave_rmae", "wave_rmse"),
}
MISSING_TOKENS = {"", "-", "na", "n/a", "none", "null"}


def _norm(raw: str) -> str:
    return (raw or "").strip().lower()


def _is_missing(raw: str) -> bool:
    return _norm(raw) in MISSING_TOKENS


def parse_float(raw: str) -> Optional[float]:
    text = (raw or "").strip()
    if _is_missing(text):
        return None
    if text.lower() == "oom":
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def pct_improve(base: Optional[float], ours: Optional[float]) -> Optional[float]:
    if base is None or ours is None or base <= 0:
        return None
    return (base - ours) / base * 100.0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_loss(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1e}"


def format_metric(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def format_percent(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}\\%"


def format_promo(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.0f}\\%"


def format_baseline_cell(raw: str, kind: str) -> str:
    text = (raw or "").strip()
    if _is_missing(text):
        return "N/A"
    if text.lower() == "oom":
        return "OOM"
    val = parse_float(text)
    if val is None:
        return text.replace("%", "\\%")
    if kind == "loss":
        return format_loss(val)
    if kind == "percent":
        return format_percent(val, digits=1)
    return format_metric(val)


def expected_metrics_path(
    results_dir: Path,
    task: str,
    ours_model: str,
    run_tag_prefix: str,
    base_model: str,
    seed: int,
    max_iters: int,
) -> Path:
    task_prefix = "1dreaction" if task == "reaction" else "1dwave"
    run_tag = f"{run_tag_prefix}_{task}_{base_model.lower()}_s{seed}_i{max_iters}"
    filename = f"{task_prefix}_{ours_model}_region_metrics_{run_tag}.csv"
    return results_dir / filename


def load_metrics_csv(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        raise FileNotFoundError(f"missing metrics file: {path}")

    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            key = (row.get("metric") or "").strip()
            raw = (row.get("value") or "").strip()
            val = parse_float(raw)
            if key and val is not None:
                metrics[key] = val

    if "relative_l1" not in metrics or "relative_l2" not in metrics:
        raise ValueError(f"missing relative_l1/relative_l2 in {path}")
    if "train_loss" not in metrics:
        keys = ("loss_0", "loss_1", "loss_2")
        if all(k in metrics for k in keys):
            metrics["train_loss"] = metrics["loss_0"] + metrics["loss_1"] + metrics["loss_2"]
        else:
            raise ValueError(f"missing train_loss and loss_0/1/2 in {path}")
    return metrics


def load_baseline(path: Path, models: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"baseline csv not found: {path}")
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            model = (row.get("model") or "").strip()
            objective = (row.get("objective") or "").strip()
            if not model or not objective:
                continue
            out[(model, objective)] = {k: (v or "").strip() for k, v in row.items()}

    for model in models:
        for obj in OBJECTIVE_ORDER:
            if (model, obj) not in out:
                raise ValueError(f"baseline csv missing row: model={model}, objective={obj}")
    return out


def build_summary_and_win(
    baseline: Dict[Tuple[str, str], Dict[str, str]],
    ours_metrics: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
    tasks: Sequence[str],
    seed: int,
    max_iters: int,
    run_tag_prefix: str,
    metrics_paths: Dict[Tuple[str, str], Path],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    win_rows: List[Dict[str, object]] = []

    for model in models:
        row: Dict[str, object] = {
            "model": model,
            "seed": seed,
            "max_iters": max_iters,
            "run_tag_prefix": run_tag_prefix,
        }
        win_all = True
        for task in tasks:
            loss_col, rmae_col, rmse_col = TASK_COLS[task]
            base_row = baseline[(model, "RoPINN")]
            base_loss = parse_float(base_row.get(loss_col, ""))
            base_rmae = parse_float(base_row.get(rmae_col, ""))
            base_rmse = parse_float(base_row.get(rmse_col, ""))

            ours = ours_metrics[model][task]
            ours_loss = ours["train_loss"]
            ours_rmae = ours["relative_l1"]
            ours_rmse = ours["relative_l2"]

            promo_loss = pct_improve(base_loss, ours_loss)
            promo_rmae = pct_improve(base_rmae, ours_rmae)
            promo_rmse = pct_improve(base_rmse, ours_rmse)

            win = bool(
                base_rmae is not None
                and base_rmse is not None
                and ours_rmae < base_rmae
                and ours_rmse < base_rmse
            )
            win_all = win_all and win

            row[f"{task}_ropinn_loss"] = base_loss
            row[f"{task}_ropinn_rmae"] = base_rmae
            row[f"{task}_ropinn_rmse"] = base_rmse
            row[f"{task}_resff_loss"] = ours_loss
            row[f"{task}_resff_rmae"] = ours_rmae
            row[f"{task}_resff_rmse"] = ours_rmse
            row[f"{task}_promotion_loss_pct"] = promo_loss
            row[f"{task}_promotion_rmae_pct"] = promo_rmae
            row[f"{task}_promotion_rmse_pct"] = promo_rmse
            row[f"win_{task}"] = "Win" if win else "Fail"
            row[f"{task}_metrics_path"] = str(metrics_paths[(model, task)])

            win_rows.append(
                {
                    "model": model,
                    "task": task,
                    "baseline_loss": base_loss,
                    "baseline_rmae": base_rmae,
                    "baseline_rmse": base_rmse,
                    "ours_loss": ours_loss,
                    "ours_rmae": ours_rmae,
                    "ours_rmse": ours_rmse,
                    "promotion_loss_pct": promo_loss,
                    "promotion_rmae_pct": promo_rmae,
                    "promotion_rmse_pct": promo_rmse,
                    "win": "Win" if win else "Fail",
                    "metrics_path": str(metrics_paths[(model, task)]),
                }
            )

        row["win_all_selected"] = "Win" if win_all else "Fail"
        summary_rows.append(row)

    return summary_rows, win_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def render_tex(
    path: Path,
    baseline: Dict[Tuple[str, str], Dict[str, str]],
    ours_metrics: Dict[str, Dict[str, Dict[str, float]]],
    models: Sequence[str],
    tasks: Sequence[str],
) -> None:
    selected = set(tasks)

    lines: List[str] = [
        r"\begin{table*}[t]",
        r"  \caption{Table-1 style comparison on 1D-Reaction and 1D-Wave using paper constants as baseline and newly measured RoPINN-ResFF runs.}",
        r"  \label{tab:ropinn-table1-rw}",
        r"  \centering",
        r"  \begin{small}",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \begin{tabular}{llcccccccc}",
        r"      \toprule",
        r"      \multirow{2}{*}{Base Model} & \multirow{2}{*}{Objective} & \multicolumn{3}{c}{1D-Reaction} & \multicolumn{3}{c}{1D-Wave} & \multicolumn{2}{c}{PINNacle (16 tasks)} \\",
        r"      \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"       &  & Loss & rMAE & rMSE & Loss & rMAE & rMSE & rMAE & rMSE \\",
        r"      \midrule",
    ]

    for m_idx, model in enumerate(models):
        model_label = MODEL_DISPLAY.get(model, model)
        ours = ours_metrics[model]
        r_ours = ours.get("reaction", {})
        w_ours = ours.get("wave", {})
        ro = baseline[(model, "RoPINN")]

        for o_idx, objective in enumerate(OBJECTIVE_ORDER):
            b = baseline[(model, objective)]
            model_cell = rf"\multirow{{6}}{{*}}{{{model_label}}}" if o_idx == 0 else ""

            r_loss = format_baseline_cell(b.get("reaction_loss", ""), "loss")
            r_rmae = format_baseline_cell(b.get("reaction_rmae", ""), "metric")
            r_rmse = format_baseline_cell(b.get("reaction_rmse", ""), "metric")
            w_loss = format_baseline_cell(b.get("wave_loss", ""), "loss")
            w_rmae = format_baseline_cell(b.get("wave_rmae", ""), "metric")
            w_rmse = format_baseline_cell(b.get("wave_rmse", ""), "metric")
            p_rmae = format_baseline_cell(b.get("pinnacle_rmae", ""), "percent")
            p_rmse = format_baseline_cell(b.get("pinnacle_rmse", ""), "percent")

            lines.append(
                f"      {model_cell} & {objective} & {r_loss} & {r_rmae} & {r_rmse} & "
                f"{w_loss} & {w_rmae} & {w_rmse} & {p_rmae} & {p_rmse} \\\\"
            )

        r_loss = format_loss(r_ours.get("train_loss") if "reaction" in selected else None)
        r_rmae = format_metric(r_ours.get("relative_l1") if "reaction" in selected else None)
        r_rmse = format_metric(r_ours.get("relative_l2") if "reaction" in selected else None)
        w_loss = format_loss(w_ours.get("train_loss") if "wave" in selected else None)
        w_rmae = format_metric(w_ours.get("relative_l1") if "wave" in selected else None)
        w_rmse = format_metric(w_ours.get("relative_l2") if "wave" in selected else None)
        lines.append(
            "       & RoPINN-ResFF & "
            f"{r_loss} & {r_rmae} & {r_rmse} & {w_loss} & {w_rmae} & {w_rmse} & N/A & N/A \\\\"
        )

        r_p_loss = pct_improve(parse_float(ro.get("reaction_loss", "")), r_ours.get("train_loss") if "reaction" in selected else None)
        r_p_rmae = pct_improve(parse_float(ro.get("reaction_rmae", "")), r_ours.get("relative_l1") if "reaction" in selected else None)
        r_p_rmse = pct_improve(parse_float(ro.get("reaction_rmse", "")), r_ours.get("relative_l2") if "reaction" in selected else None)
        w_p_loss = pct_improve(parse_float(ro.get("wave_loss", "")), w_ours.get("train_loss") if "wave" in selected else None)
        w_p_rmae = pct_improve(parse_float(ro.get("wave_rmae", "")), w_ours.get("relative_l1") if "wave" in selected else None)
        w_p_rmse = pct_improve(parse_float(ro.get("wave_rmse", "")), w_ours.get("relative_l2") if "wave" in selected else None)
        lines.append(
            "       & Promotion & "
            f"{format_promo(r_p_loss)} & {format_promo(r_p_rmae)} & {format_promo(r_p_rmse)} & "
            f"{format_promo(w_p_loss)} & {format_promo(w_p_rmae)} & {format_promo(w_p_rmse)} & N/A & N/A \\\\"
        )

        if m_idx < len(models) - 1:
            lines.append(r"      \midrule")

    lines.extend(
        [
            r"      \bottomrule",
            r"    \end{tabular}",
            r"  \end{small}",
            r"\end{table*}",
            "",
        ]
    )

    ensure_parent(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Build Table-1 style Reaction/Wave artifacts for RoPINN-ResFF.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root_dir / "results",
        help="Directory where 1dreaction_* and 1dwave_* metrics CSV files are written.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=root_dir / "scripts" / "table1_resff_rw_baseline.csv",
        help="Paper baseline constants CSV.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS_ALL,
        default=list(TASKS_ALL),
        help="Tasks to include for ours runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_TO_OURS.keys()),
        default=list(MODEL_TO_OURS.keys()),
        help="Base models to include.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used in run tags.")
    parser.add_argument("--max-iters", type=int, default=1000, help="Max iterations used in run tags.")
    parser.add_argument(
        "--run-tag-prefix",
        type=str,
        default="table1rw",
        help="Prefix used by run script for --run_tag.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=root_dir / "results" / "table1_resff_rw" / "summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--out-win",
        type=Path,
        default=root_dir / "results" / "table1_resff_rw" / "win_loss.csv",
        help="Output Win/Loss CSV path.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=root_dir / "paper" / "tables" / "table_ropinn_table1_plus_ours_rw.tex",
        help="Output LaTeX table path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(args.models)
    tasks = list(args.tasks)
    baseline = load_baseline(args.baseline_csv, models)

    ours_metrics: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in models}
    metrics_paths: Dict[Tuple[str, str], Path] = {}

    for model in models:
        ours_model = MODEL_TO_OURS[model]
        for task in tasks:
            m_path = expected_metrics_path(
                results_dir=args.results_dir,
                task=task,
                ours_model=ours_model,
                run_tag_prefix=args.run_tag_prefix,
                base_model=model,
                seed=args.seed,
                max_iters=args.max_iters,
            )
            metrics_paths[(model, task)] = m_path
            ours_metrics[model][task] = load_metrics_csv(m_path)

    summary_rows, win_rows = build_summary_and_win(
        baseline=baseline,
        ours_metrics=ours_metrics,
        models=models,
        tasks=tasks,
        seed=args.seed,
        max_iters=args.max_iters,
        run_tag_prefix=args.run_tag_prefix,
        metrics_paths=metrics_paths,
    )

    summary_fields: List[str] = ["model", "seed", "max_iters", "run_tag_prefix"]
    for task in tasks:
        summary_fields.extend(
            [
                f"{task}_ropinn_loss",
                f"{task}_ropinn_rmae",
                f"{task}_ropinn_rmse",
                f"{task}_resff_loss",
                f"{task}_resff_rmae",
                f"{task}_resff_rmse",
                f"{task}_promotion_loss_pct",
                f"{task}_promotion_rmae_pct",
                f"{task}_promotion_rmse_pct",
                f"win_{task}",
                f"{task}_metrics_path",
            ]
        )
    summary_fields.append("win_all_selected")

    win_fields = [
        "model",
        "task",
        "baseline_loss",
        "baseline_rmae",
        "baseline_rmse",
        "ours_loss",
        "ours_rmae",
        "ours_rmse",
        "promotion_loss_pct",
        "promotion_rmae_pct",
        "promotion_rmse_pct",
        "win",
        "metrics_path",
    ]

    write_csv(args.out_csv, summary_rows, summary_fields)
    write_csv(args.out_win, win_rows, win_fields)
    render_tex(args.out_tex, baseline, ours_metrics, models, tasks)

    print("[OK] Artifacts generated:")
    print(f"- summary csv: {args.out_csv}")
    print(f"- win/loss csv: {args.out_win}")
    print(f"- latex table: {args.out_tex}")


if __name__ == "__main__":
    main()
