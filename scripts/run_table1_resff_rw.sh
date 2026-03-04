#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
TABLE_DIR="${RESULTS_DIR}/table1_resff_rw"
LOG_DIR="${TABLE_DIR}/logs"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-table1rw}"

FF_DIM_REACTION="${FF_DIM_REACTION:-64}"
FF_SCALE_REACTION="${FF_SCALE_REACTION:-0.5}"
FF_DIM_WAVE="${FF_DIM_WAVE:-64}"
FF_SCALE_WAVE="${FF_SCALE_WAVE:-1.0}"

# Override when debugging:
# TASKS="reaction wave"
# MODELS="PINN QRes FLS PINNsFormer KAN"
TASKS="${TASKS:-reaction wave}"
MODELS="${MODELS:-PINN QRes FLS PINNsFormer KAN}"

BASELINE_CSV="${BASELINE_CSV:-${ROOT_DIR}/scripts/table1_resff_rw_baseline.csv}"
OUT_CSV="${OUT_CSV:-${TABLE_DIR}/summary.csv}"
OUT_WIN="${OUT_WIN:-${TABLE_DIR}/win_loss.csv}"
OUT_TEX="${OUT_TEX:-${ROOT_DIR}/paper/tables/table_ropinn_table1_plus_ours_rw.tex}"

map_model() {
  local base_model="$1"
  case "${base_model}" in
    PINN) echo "PINN_ResFF" ;;
    QRes) echo "QRes_FF" ;;
    FLS) echo "FLS_FF" ;;
    PINNsFormer) echo "PINNsFormer_FF" ;;
    KAN) echo "KAN_FF" ;;
    *) return 1 ;;
  esac
}

mkdir -p "${TABLE_DIR}" "${LOG_DIR}"

if [[ ! -f "${BASELINE_CSV}" ]]; then
  echo "[ERROR] Baseline CSV not found: ${BASELINE_CSV}"
  exit 1
fi

cp -f "${BASELINE_CSV}" "${TABLE_DIR}/baseline_paper.csv"

read -r -a TASK_LIST <<< "${TASKS}"
read -r -a MODEL_LIST <<< "${MODELS}"

total=$(( ${#TASK_LIST[@]} * ${#MODEL_LIST[@]} ))
idx=0

run_case() {
  local task="$1"
  local base_model="$2"
  local ours_model=""
  local task_script=""
  local run_tag=""
  local model_slug=""
  local log_file=""
  local -a cmd=()

  if ! ours_model="$(map_model "${base_model}")"; then
    echo "[ERROR] Unsupported model '${base_model}'."
    exit 1
  fi

  model_slug="$(echo "${base_model}" | tr '[:upper:]' '[:lower:]')"
  run_tag="${RUN_TAG_PREFIX}_${task}_${model_slug}_s${SEED}_i${MAX_ITERS}"
  log_file="${LOG_DIR}/${run_tag}.log"

  case "${task}" in
    reaction)
      task_script="${ROOT_DIR}/1d_reaction_region_optimization.py"
      cmd=(
        python -u "${task_script}"
        --model "${ours_model}"
        --device "${DEVICE}"
        --seed "${SEED}"
        --max_iters "${MAX_ITERS}"
        --ff_dim "${FF_DIM_REACTION}"
        --ff_scale "${FF_SCALE_REACTION}"
        --paper_outputs
        --run_tag "${run_tag}"
      )
      ;;
    wave)
      task_script="${ROOT_DIR}/1d_wave_region_optimization.py"
      cmd=(
        python -u "${task_script}"
        --model "${ours_model}"
        --device "${DEVICE}"
        --seed "${SEED}"
        --max_iters "${MAX_ITERS}"
        --ff_dim "${FF_DIM_WAVE}"
        --ff_scale "${FF_SCALE_WAVE}"
        --paper_outputs
        --run_tag "${run_tag}"
      )
      ;;
    *)
      echo "[ERROR] Unsupported task '${task}'. Use: reaction wave"
      exit 1
      ;;
  esac

  idx=$((idx + 1))
  echo
  echo "================ [${idx}/${total}] ${task} | ${base_model} -> ${ours_model} ================"
  echo -n "CMD: "
  printf "%q " "${cmd[@]}"
  echo
  echo "LOG: ${log_file}"
  echo "-----------------------------------------------------------"

  PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPL_DIR}" "${cmd[@]}" 2>&1 | tee "${log_file}"

  local l1 l2
  l1=$(grep "relative L1 error" "${log_file}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${log_file}" | tail -1 | awk '{print $4}')
  echo "[DONE] ${task}/${base_model}  L1=${l1:-NA}  L2=${l2:-NA}"
}

for task in "${TASK_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    run_case "${task}" "${model}"
  done
done

python "${ROOT_DIR}/scripts/build_table1_resff_rw.py" \
  --results-dir "${RESULTS_DIR}" \
  --baseline-csv "${BASELINE_CSV}" \
  --tasks "${TASK_LIST[@]}" \
  --models "${MODEL_LIST[@]}" \
  --seed "${SEED}" \
  --max-iters "${MAX_ITERS}" \
  --run-tag-prefix "${RUN_TAG_PREFIX}" \
  --out-csv "${OUT_CSV}" \
  --out-win "${OUT_WIN}" \
  --out-tex "${OUT_TEX}"

echo
echo "[OK] Finished Table-1 RW pipeline."
echo "- Baseline constants: ${TABLE_DIR}/baseline_paper.csv"
echo "- Summary CSV: ${OUT_CSV}"
echo "- Win/Loss CSV: ${OUT_WIN}"
echo "- LaTeX table: ${OUT_TEX}"
