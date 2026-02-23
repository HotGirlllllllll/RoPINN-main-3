#!/usr/bin/env bash
set -euo pipefail

# Extend multi-seed experiments and refresh paper tables/stat tests.
#
# Default behavior:
# - tasks: reaction wave convection
# - seeds: 5..9
# - baseline: backup directory "RoPINN-main 3"
# - ours: current branch PINN_ResFF
#
# Example:
#   TASKS="reaction" SEED_START=5 SEED_END=14 DEVICE=cuda:0 \
#   BASELINE_DIR="/root/autodl-tmp/RoPINN-main-3/RoPINN-main 3" \
#   bash scripts/paper_extend_multiseed.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/RoPINN-main 3}"
DEVICE="${DEVICE:-cuda:0}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"
SEED_START="${SEED_START:-5}"
SEED_END="${SEED_END:-9}"
MAX_ITERS="${MAX_ITERS:-1000}"
TASKS="${TASKS:-reaction wave convection}"
TMP_DIR="${TMP_DIR:-/tmp/ropinn_seeded}"
REFRESH_TABLES="${REFRESH_TABLES:-1}"
BUILD_PAPER="${BUILD_PAPER:-0}"

mkdir -p "${RESULTS_DIR}" "${TMP_DIR}"

make_seeded_copy() {
  local src="$1"
  local dst="$2"
  local seed="$3"
  python - "$src" "$dst" "$seed" <<'PY'
from pathlib import Path
import re
import sys

src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = Path(src).read_text(encoding="utf-8")
new_text, n = re.subn(r"^seed\s*=\s*\d+\s*$", f"seed = {seed}", text, flags=re.M)
if n == 0:
    raise SystemExit(f"[ERROR] cannot find global seed assignment in {src}")
Path(dst).write_text(new_text, encoding="utf-8")
print(dst)
PY
}

extract_metric() {
  local log="$1"
  local key="$2"
  grep "${key}" "${log}" | tail -1 | awk '{print $4}'
}

run_one() {
  local task="$1"
  local seed="$2"

  local baseline_src ours_src prefix ours_extra
  case "${task}" in
    reaction)
      baseline_src="${BASELINE_DIR}/1d_reaction_region_optimization.py"
      ours_src="${ROOT_DIR}/1d_reaction_region_optimization.py"
      prefix="react"
      ours_extra=(--ff_dim 64 --ff_scale 0.5)
      ;;
    wave)
      baseline_src="${BASELINE_DIR}/1d_wave_region_optimization.py"
      ours_src="${ROOT_DIR}/1d_wave_region_optimization.py"
      prefix="wave"
      ours_extra=()
      ;;
    convection)
      baseline_src="${BASELINE_DIR}/convection_region_optimization.py"
      ours_src="${ROOT_DIR}/convection_region_optimization.py"
      prefix="conv"
      ours_extra=()
      ;;
    *)
      echo "[ERROR] unknown task: ${task}" >&2
      exit 1
      ;;
  esac

  local baseline_py="${TMP_DIR}/${task}_base_s${seed}.py"
  local base_log="${RESULTS_DIR}/mseed_${prefix}_base_s${seed}.log"
  local ours_log="${RESULTS_DIR}/mseed_${prefix}_resff_s${seed}.log"

  make_seeded_copy "${baseline_src}" "${baseline_py}" "${seed}" >/dev/null

  echo
  echo "================ [${task}] seed ${seed} baseline ================"
  PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPL_DIR}" PYTHONPATH="${BASELINE_DIR}" \
    python -u "${baseline_py}" \
      --model PINN --device "${DEVICE}" --max_iters "${MAX_ITERS}" \
      2>&1 | tee "${base_log}"

  echo
  echo "================ [${task}] seed ${seed} ours ===================="
  PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPL_DIR}" PYTHONPATH="${ROOT_DIR}" \
    python -u "${ours_src}" \
      --model PINN_ResFF --device "${DEVICE}" --seed "${seed}" --max_iters "${MAX_ITERS}" \
      "${ours_extra[@]}" \
      2>&1 | tee "${ours_log}"

  local b_l1 b_l2 o_l1 o_l2
  b_l1="$(extract_metric "${base_log}" "relative L1 error" || true)"
  b_l2="$(extract_metric "${base_log}" "relative L2 error" || true)"
  o_l1="$(extract_metric "${ours_log}" "relative L1 error" || true)"
  o_l2="$(extract_metric "${ours_log}" "relative L2 error" || true)"
  echo "[DONE] ${task} s${seed}  baseline(L1=${b_l1:-NA}, L2=${b_l2:-NA})  ours(L1=${o_l1:-NA}, L2=${o_l2:-NA})"
}

echo "ROOT_DIR=${ROOT_DIR}"
echo "BASELINE_DIR=${BASELINE_DIR}"
echo "DEVICE=${DEVICE}"
echo "TASKS=${TASKS}"
echo "SEEDS=${SEED_START}..${SEED_END}"
echo "MAX_ITERS=${MAX_ITERS}"

for s in $(seq "${SEED_START}" "${SEED_END}"); do
  for task in ${TASKS}; do
    run_one "${task}" "${s}"
  done
done

if [[ "${REFRESH_TABLES}" == "1" ]]; then
  echo
  echo "================ Refresh paper tables/stat ======================"
  python "${ROOT_DIR}/scripts/paper_make_tables.py" --strict
  python "${ROOT_DIR}/scripts/paper_significance_tests.py"
fi

if [[ "${BUILD_PAPER}" == "1" ]]; then
  if command -v latexmk >/dev/null 2>&1; then
    echo
    echo "================ Build paper/main.pdf ==========================="
    (cd "${ROOT_DIR}/paper" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
  else
    echo "[WARN] BUILD_PAPER=1 but latexmk is not available in PATH."
  fi
fi

echo
echo "All done."
