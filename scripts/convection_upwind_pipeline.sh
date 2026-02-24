#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
DEVICE="${DEVICE:-cuda:0}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"

# Stage-1: fast alpha search on one seed
QUICK_SEED="${QUICK_SEED:-1}"
QUICK_ITERS="${QUICK_ITERS:-100}"
ALPHAS="${ALPHAS:-0.0 0.2 0.5 0.8}"

# Stage-2: full verification on multiple seeds
FULL_ITERS="${FULL_ITERS:-1000}"
FULL_SEEDS="${FULL_SEEDS:-0 1 2}"

mkdir -p "${RESULTS_DIR}"

COMMON_ARGS=(
  --model PINN_ResFF
  --device "${DEVICE}"
  --sampling_mode one_sided
  --sample_num 4
  --residual_loss mse
  --use_characteristic
  --adv_speed 50
  --ff_basis gaussian
  --ff_seed 0
  --ff_dim 64
  --ff_scale 1.0
  --ff_scale_x 0.1
  --ff_scale_t 0.01
  --ff_scale_char 0.5
  --w_res 1 --w_bc 1 --w_ic 5
  --paper_outputs
)

run_case() {
  local tag="$1"
  shift
  local log="${RESULTS_DIR}/${tag}.log"
  echo
  echo "================ ${tag} ================"
  echo "LOG: ${log}"
  PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPL_DIR}" \
    python -u "${ROOT_DIR}/convection_region_optimization.py" "$@" \
    2>&1 | tee "${log}"
  local l1 l2
  l1=$(grep "relative L1 error" "${log}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${log}" | tail -1 | awk '{print $4}')
  echo "[DONE] ${tag}  L1=${l1:-NA}  L2=${l2:-NA}"
}

echo "ROOT_DIR=${ROOT_DIR}"
echo "DEVICE=${DEVICE}"
echo "MPL_DIR=${MPL_DIR}"
echo "QUICK_SEED=${QUICK_SEED}, QUICK_ITERS=${QUICK_ITERS}, ALPHAS=${ALPHAS}"
echo "FULL_SEEDS=${FULL_SEEDS}, FULL_ITERS=${FULL_ITERS}"

echo
echo "######## Stage-1: Quick alpha search ########"
for alpha in ${ALPHAS}; do
  tag="conv_upwind_quick_s${QUICK_SEED}_a${alpha/./p}"
  run_case "${tag}" \
    "${COMMON_ARGS[@]}" \
    --seed "${QUICK_SEED}" \
    --max_iters "${QUICK_ITERS}" \
    --upwind_bias_alpha "${alpha}" \
    --upwind_bias_time_scale 1.0 \
    --run_tag "${tag}"
done

BEST_ALPHA=$(python - <<'PY'
import glob, re
p2 = re.compile(r"relative L2 error:\s*([0-9eE+.-]+)")
best = None
for f in sorted(glob.glob("results/conv_upwind_quick_s*_a*.log")):
    txt = open(f, encoding="utf-8", errors="ignore").read()
    m = p2.findall(txt)
    if not m:
        continue
    l2 = float(m[-1])
    alpha = f.split("_a")[-1].split(".log")[0].replace("p", ".")
    if best is None or l2 < best[1]:
        best = (alpha, l2, f)
if best is None:
    raise SystemExit("No valid quick-search logs found.")
print(best[0])
PY
)

echo
echo "[SELECTED] best upwind_bias_alpha=${BEST_ALPHA}"

echo
echo "######## Stage-2: Full runs with selected alpha ########"
for seed in ${FULL_SEEDS}; do
  tag="conv_upwind_full_s${seed}_a${BEST_ALPHA/./p}"
  run_case "${tag}" \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --max_iters "${FULL_ITERS}" \
    --upwind_bias_alpha "${BEST_ALPHA}" \
    --upwind_bias_time_scale 1.0 \
    --run_tag "${tag}"
done

python - <<'PY'
import glob, re, statistics as st
p1 = re.compile(r"relative L1 error:\s*([0-9eE+.-]+)")
p2 = re.compile(r"relative L2 error:\s*([0-9eE+.-]+)")
rows = []
for f in sorted(glob.glob("results/conv_upwind_full_s*_a*.log")):
    t = open(f, encoding="utf-8", errors="ignore").read()
    a = float(p1.findall(t)[-1]); b = float(p2.findall(t)[-1])
    rows.append((f, a, b))
print("\n================ Convection Upwind Summary ================")
for f, a, b in rows:
    print(f"{f}  L1={a:.6f}  L2={b:.6f}")
l1 = [x[1] for x in rows]; l2 = [x[2] for x in rows]
print(f"mean±std: L1={st.mean(l1):.6f}±{st.pstdev(l1):.6f}, L2={st.mean(l2):.6f}±{st.pstdev(l2):.6f}")
PY

