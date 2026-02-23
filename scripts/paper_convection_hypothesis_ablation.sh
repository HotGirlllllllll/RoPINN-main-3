#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results/conv_hypothesis"
DEVICE="${DEVICE:-cuda:0}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/mpl}"
MAX_ITERS="${MAX_ITERS:-1000}"

mkdir -p "${RESULTS_DIR}"

run_case() {
  local tag="$1"
  local cmd="$2"
  local log="${RESULTS_DIR}/${tag}.log"
  echo
  echo "================ ${tag} ================"
  echo "CMD: ${cmd}"
  echo "LOG: ${log}"
  eval "PYTHONUNBUFFERED=1 MPLCONFIGDIR=${MPL_DIR} ${cmd}" 2>&1 | tee "${log}"
  local l1 l2
  l1=$(grep "relative L1 error" "${log}" | tail -1 | awk '{print $4}')
  l2=$(grep "relative L2 error" "${log}" | tail -1 | awk '{print $4}')
  echo "[DONE] ${tag} L1=${l1:-NA} L2=${l2:-NA}"
}

for seed in 0 1 2; do
  run_case "conv_resff_iso_s${seed}" \
    "python -u \"${ROOT_DIR}/convection_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${seed} --max_iters ${MAX_ITERS} --ff_dim 64 --ff_scale 1.0"

  run_case "conv_resff_aniso_s${seed}" \
    "python -u \"${ROOT_DIR}/convection_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${seed} --max_iters ${MAX_ITERS} --ff_dim 64 --ff_scale 1.0 --ff_scale_x 1.0 --ff_scale_t 0.1"

  run_case "conv_resff_char_s${seed}" \
    "python -u \"${ROOT_DIR}/convection_region_optimization.py\" --model PINN_ResFF --device ${DEVICE} --seed ${seed} --max_iters ${MAX_ITERS} --ff_dim 64 --ff_scale 1.0 --ff_scale_x 1.0 --ff_scale_t 0.1 --use_characteristic --adv_speed 50.0 --ff_scale_char 1.0"
done

python - <<'PY'
import glob, re, statistics as st
p1=re.compile(r"relative L1 error:\s*([0-9eE+.-]+)")
p2=re.compile(r"relative L2 error:\s*([0-9eE+.-]+)")
groups = [
    ("iso", "results/conv_hypothesis/conv_resff_iso_s*.log"),
    ("aniso", "results/conv_hypothesis/conv_resff_aniso_s*.log"),
    ("char", "results/conv_hypothesis/conv_resff_char_s*.log"),
]
print("\n================ Convection Hypothesis Summary ================")
for name,pat in groups:
    l1=[]; l2=[]
    for f in sorted(glob.glob(pat)):
        t=open(f,encoding="utf-8",errors="ignore").read()
        l1.append(float(p1.findall(t)[-1])); l2.append(float(p2.findall(t)[-1]))
    print(f"{name:6s}  L1={st.mean(l1):.6f}±{st.pstdev(l1):.6f}  L2={st.mean(l2):.6f}±{st.pstdev(l2):.6f}")
PY

