#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/RoPINN-main 3}"
DEVICE="${DEVICE:-cuda:0}"
MAX_ITERS="${MAX_ITERS:-1000}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"

mkdir -p "${RESULTS_DIR}"

total=4
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

run_case "paper_best_reaction_1000" \
"python -u \"${ROOT_DIR}/1d_reaction_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --max_iters ${MAX_ITERS} --ff_dim 64 --ff_scale 0.5 --use_curriculum --curriculum_switch_ratio 0.7 --curriculum_stage1_loss mse --curriculum_stage1_sampling one_sided --curriculum_stage1_sample_num 1 --curriculum_stage2_loss mse --curriculum_stage2_sampling one_sided --curriculum_stage2_sample_num 6 --paper_outputs --run_tag paper_best"

echo
echo "================ Summary ================"
for f in "${RESULTS_DIR}"/paper_*_1000.log; do
  tag=$(basename "${f}" .log)
  l1=$(grep "relative L1 error" "${f}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${f}" | tail -1 | awk '{print $4}')
  echo "${tag}  L1=${l1:-NA}  L2=${l2:-NA}"
done

python "${ROOT_DIR}/scripts/paper_collect_results.py" \
  --glob "${RESULTS_DIR}/paper_*_1000.log" \
  --outdir "${RESULTS_DIR}/paper" \
  --baseline "paper_base_reaction_1000"

echo
echo "Paper artifacts generated under: ${RESULTS_DIR}/paper"
echo "- summary.csv"
echo "- summary.md"
echo "- summary_bar.pdf"
echo "- reduction_vs_baseline.csv"
