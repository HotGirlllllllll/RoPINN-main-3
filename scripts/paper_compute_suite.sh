#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results/paper/compute"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/RoPINN-main 3}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
MAX_ITERS="${MAX_ITERS:-1000}"
FF_DIM="${FF_DIM:-64}"
FF_SCALE="${FF_SCALE:-0.5}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"
RESET_CSV="${RESET_CSV:-0}"
RUN_CAPMATCH="${RUN_CAPMATCH:-1}"

mkdir -p "${RESULTS_DIR}"

CSV_PATH="${RESULTS_DIR}/compute_runs.csv"
if [[ "${RESET_CSV}" == "1" ]]; then
  rm -f "${CSV_PATH}"
fi
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "task,method,seed,max_iters,elapsed_sec,sec_per_iter,params,relative_l1,relative_l2,log_path" > "${CSV_PATH}"
fi

run_case() {
  local task="$1"
  local method="$2"
  local max_iters="$3"
  local cmd="$4"
  local tag="compute_${task}_${method}_s${SEED}"
  local log="${RESULTS_DIR}/${tag}.log"

  echo
  echo "================ ${tag} ================"
  echo "CMD: ${cmd}"
  echo "LOG: ${log}"
  echo "----------------------------------------"

  local t_start t_end elapsed sec_per_iter l1 l2 params
  t_start=$(python - <<'PY'
import time
print(f"{time.time():.9f}")
PY
)

  eval "PYTHONUNBUFFERED=1 MPLCONFIGDIR=${MPL_DIR} ${cmd}" 2>&1 | tee "${log}"

  t_end=$(python - <<'PY'
import time
print(f"{time.time():.9f}")
PY
)

  elapsed=$(python - "${t_start}" "${t_end}" <<'PY'
import sys
t0 = float(sys.argv[1])
t1 = float(sys.argv[2])
print(f"{max(0.0, t1 - t0):.6f}")
PY
)

  sec_per_iter=$(python - "${elapsed}" "${max_iters}" <<'PY'
import sys
elapsed = float(sys.argv[1])
iters = int(sys.argv[2])
print(f"{(elapsed / iters) if iters > 0 else 0.0:.6f}")
PY
)

  l1=$(grep "relative L1 error" "${log}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${log}" | tail -1 | awk '{print $4}')
  params=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' "${log}" | head -1 | tr -d ' ' || true)

  echo "${task},${method},${SEED},${max_iters},${elapsed},${sec_per_iter},${params:-NA},${l1:-NA},${l2:-NA},${log}" >> "${CSV_PATH}"
  echo "[DONE] ${tag} elapsed=${elapsed}s sec/iter=${sec_per_iter} params=${params:-NA} L1=${l1:-NA} L2=${l2:-NA}"
}

# Reaction
run_case "reaction" "baseline" "${MAX_ITERS}" \
"python -u \"${BASELINE_DIR}/1d_reaction_region_optimization.py\" --model PINN --device ${DEVICE}"
if [[ "${RUN_CAPMATCH}" == "1" ]]; then
  run_case "reaction" "baseline_capmatch" "${MAX_ITERS}" \
  "python -u \"${ROOT_DIR}/1d_reaction_region_optimization.py\" --model PINN --device ${DEVICE} --seed ${SEED} --max_iters ${MAX_ITERS} --match_pinn_to_resff --pinn_num_layer 4 --ff_dim ${FF_DIM} --ff_scale ${FF_SCALE}"
fi
run_case "reaction" "ours" "${MAX_ITERS}" \
"python -u \"${ROOT_DIR}/1d_reaction_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${SEED} --max_iters ${MAX_ITERS} --ff_dim ${FF_DIM} --ff_scale ${FF_SCALE}"

# Wave
run_case "wave" "baseline" "${MAX_ITERS}" \
"python -u \"${BASELINE_DIR}/1d_wave_region_optimization.py\" --model PINN --device ${DEVICE}"
run_case "wave" "ours" "${MAX_ITERS}" \
"python -u \"${ROOT_DIR}/1d_wave_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${SEED} --max_iters ${MAX_ITERS}"

# Convection
run_case "convection" "baseline" "${MAX_ITERS}" \
"python -u \"${BASELINE_DIR}/convection_region_optimization.py\" --model PINN --device ${DEVICE}"
run_case "convection" "ours" "${MAX_ITERS}" \
"python -u \"${ROOT_DIR}/convection_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${SEED} --max_iters ${MAX_ITERS}"

echo
echo "[OK] Compute logs + CSV generated:"
echo "  - ${CSV_PATH}"
echo "  - ${RESULTS_DIR}/compute_*.log"

python "${ROOT_DIR}/scripts/paper_make_compute_table.py" \
  --input-csv "${CSV_PATH}" \
  --out-csv "${ROOT_DIR}/results/paper/tables/compute_summary.csv" \
  --out-tex "${ROOT_DIR}/paper/tables/table_compute_accounting.tex"
