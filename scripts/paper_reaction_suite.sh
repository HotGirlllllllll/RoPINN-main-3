#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/RoPINN-main 3}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERS="${MAX_ITERS:-1000}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"

mkdir -p "${RESULTS_DIR}"
rm -f "${RESULTS_DIR}/paper_best_reaction_1000.log"

total=3
idx=0

run_case() {
  local tag="$1"
  local cmd="$2"
  local log="${RESULTS_DIR}/${tag}.log"
  idx=$((idx + 1))

  echo
  echo "================ [${idx}/${total}] ${tag} ================"
  echo "CMD: ${cmd}"
  echo "LOG: ${log}"
  echo "-----------------------------------------------------------"

  eval "PYTHONUNBUFFERED=1 MPLCONFIGDIR=${MPL_DIR} ${cmd}" 2>&1 | tee "${log}"

  local l1 l2
  l1=$(grep "relative L1 error" "${log}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${log}" | tail -1 | awk '{print $4}')
  echo "[DONE] ${tag}  L1=${l1:-NA}  L2=${l2:-NA}"
}

run_case "paper_base_reaction_1000" \
"python -u \"${BASELINE_DIR}/1d_reaction_region_optimization.py\" --model PINN --device ${DEVICE}"

run_case "paper_curr_pinn_reaction_1000" \
"python -u \"${ROOT_DIR}/1d_reaction_region_optimization.py\" --model PINN --device ${DEVICE} --max_iters ${MAX_ITERS} --paper_outputs --run_tag paper_curr_pinn"

run_case "paper_ablate_resff_only_1000" \
"python -u \"${ROOT_DIR}/1d_reaction_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --max_iters ${MAX_ITERS} --ff_dim 64 --ff_scale 0.5 --paper_outputs --run_tag paper_resff_only"

echo
echo "================ Summary ================"
for f in \
  "${RESULTS_DIR}/paper_base_reaction_1000.log" \
  "${RESULTS_DIR}/paper_curr_pinn_reaction_1000.log" \
  "${RESULTS_DIR}/paper_ablate_resff_only_1000.log"; do
  [[ -f "${f}" ]] || continue
  tag=$(basename "${f}" .log)
  l1=$(grep "relative L1 error" "${f}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${f}" | tail -1 | awk '{print $4}')
  echo "${tag}  L1=${l1:-NA}  L2=${l2:-NA}"
done

python "${ROOT_DIR}/scripts/paper_collect_results.py" \
  --glob "${RESULTS_DIR}/paper_*_reaction_1000.log" \
  --outdir "${RESULTS_DIR}/paper" \
  --baseline "paper_base_reaction_1000"

echo
echo "Paper artifacts generated under: ${RESULTS_DIR}/paper"
echo "- summary.csv"
echo "- summary.md"
echo "- summary_bar.pdf"
echo "- reduction_vs_baseline.csv"
