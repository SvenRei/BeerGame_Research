#!/usr/bin/env bash
# pod_sweep.sh -- SIGNAL v6 confirmatory campaign orchestrator (Linux pod).
#
# Staged, idempotent, fail-closed:
#   0 gauntlet    all three test suites must pass or nothing runs
#   1 scales      R9 divisors measured per (family,rho) BEFORE any training
#   2 baselines   StaticBS/CondBS with per-episode dumps, per (family,rho)
#   3 train       declared cell matrix, WORKERS-parallel, skip-if-done
#   4 eval        honest + zeroed + shuffled per comm arm (+ honest for nocomm)
#   5 analyze     report + stats per family/rho + curves
#   6 hypotheses  registered rules self-test + campaign assembly stub
#   7 manifest    sha256 of code+configs, arm inventory, environment record
#
# Idempotency: a training cell is DONE iff runs/logs/<tag>.log contains
# "[signal] done." AND runs/<tag>/ckpt_best.pt exists. Re-running the script only
# fills gaps. Evals skip when their canonical dump exists.
#
# CAPABILITY FAIL-CLOSE: cells whose content/regime/topology is not implemented in
# this build abort the run at plan time with the missing-capability manifest, rather
# than silently running a different experiment.
#
# Usage:
#   ./pod_sweep.sh plan            # print the job matrix + estimates, run nothing
#   ./pod_sweep.sh run             # execute all stages
#   WORKERS=27 SEED_START=30 N_SEEDS=15 ./pod_sweep.sh run
set -euo pipefail
cd "$(dirname "$0")"

# ------------------------------------------------------------------ parameters
WORKERS="${WORKERS:-8}"
SEED_START="${SEED_START:-30}"          # confirmatory space per skeleton SS6
N_SEEDS="${N_SEEDS:-15}"                # n from the power protocol; fallback 25
EPISODES="${EPISODES:-24000}"
PY="${PY:-python3}"

# ------------------------------------------------------------------ declared cells
# Each line: family|rho|content|topology|beta|clip   ("-" = no garbling)
DR_LO="${DR_LO:-4}"; DR_HI="${DR_HI:-24}"
CELLS=()
# F_CONTENT ladder @ rho .9 (retailer_broadcast, beta 1)
for c in nocomm raw arpred dhatc ip learned; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|-"); done
# P1 (DP side): regime uncertainty, forecast (arpred) vs raw vs nocomm.
# rho label -1 keeps DP files/stats DISJOINT from the AR rho=0 group -- otherwise the
# analysis stage would pair DP arms against AR baselines on the shared label.
for c in nocomm raw arpred; do
  CELLS+=("dr_poisson|-1|$c|retailer_broadcast|1.0|-"); done
# P2 (garbling): clip in {12, 20}; nocomm AND raw at each level (V is within-clip)
for cl in 12 20; do for c in nocomm raw; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|$cl"); done; done
# F_GEOMETRY on raw @ rho .9: positive (upstream_only == F_CONTENT raw, reused) +
# placebos; no_neighbor is the harness placebo
for t in downstream_only manufacturer_broadcast no_neighbor; do
  CELLS+=("ar1|0.9|raw|$t|1.0|-"); done
# H-TIME: staleness gradient (raw itself is lag 0, reused from F_CONTENT)
for c in raw_lag1 raw_lag2; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|-"); done
# H2 rho-gradient on raw with matched nocomm (per-rho pairing partner) + dhatc contrast
for r in 0 0.3 0.6; do for c in nocomm raw dhatc; do
  CELLS+=("ar1|$r|$c|retailer_broadcast|1.0|-"); done; done
# F_INCENTIVE: dhatc-only vs matched-beta nocomm
for b in 0 0.5; do for c in nocomm dhatc; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|$b|-"); done; done
# --- still outside this build (plan-time fail-close names them):
#UNSUPPORTED: content eps,dhat_ip,true_lambda      (messages.py rungs)
#UNSUPPORTED: QMIX second learner                  (legacy harness or descope)

SUPPORTED_CONTENT="nocomm raw arpred dhatc ip learned raw_lag1 raw_lag2"
SUPPORTED_TOPO="retailer_broadcast neighbor upstream_only downstream_only manufacturer_broadcast no_neighbor"
SUPPORTED_FAMILY="ar1 poisson dr_poisson"

tag_of() { local f=$1 r=$2 c=$3 t=$4 b=$5 s=$6 cl=$7
  local base="C_${f}_r${r//./}_${c}_${t:0:4}_b${b//./}"
  [[ "$cl" != "-" ]] && base="${base}_cl${cl}"
  echo "${base}_s${s}"; }

check_supported() { local f=$1 r=$2 c=$3 t=$4
  [[ "$f" == "dr_poisson" && "$c" == "dhatc" ]] && { echo "FAIL-CLOSED: dhatc is AR(1)-certified only"; exit 2; }
  [[ " $SUPPORTED_FAMILY " == *" $f "* ]] || { echo "FAIL-CLOSED: family '$f' not in this build"; exit 2; }
  [[ " $SUPPORTED_CONTENT " == *" $c "* ]] || { echo "FAIL-CLOSED: content '$c' not in this build"; exit 2; }
  [[ " $SUPPORTED_TOPO " == *" $t "* ]]   || { echo "FAIL-CLOSED: topology '$t' not in this build"; exit 2; }
}

scale_for() { # family rho content -> divisor from runs/msg_scales.json (fail-closed)
  $PY - "$1" "$2" "$3" <<'PYEOF'
import json, sys
fam, rho, content = sys.argv[1], sys.argv[2], sys.argv[3]
key = (f"{fam}|rho{float(rho):g}" if fam == "ar1"
       else f"dr_poisson|{sys.argv[4]}-{sys.argv[5]}" if fam == "dr_poisson"
       else f"{fam}|{rho}")
t = json.load(open("runs/msg_scales.json"))
if key not in t or content not in t[key]:
    sys.exit(f"FAIL-CLOSED: no registered divisor for ({key},{content}) -- run stage 1")
print(t[key][content])
PYEOF
}

# ------------------------------------------------------------------ plan
RHOS_NEEDED=$(printf '%s\n' "${CELLS[@]}" | awk -F'|' '$1=="ar1"{print $2}' | sort -u | paste -sd, -)
NEED_DP=$(printf '%s\n' "${CELLS[@]}" | awk -F'|' '$1=="dr_poisson"' | head -1)
N_CELLS=${#CELLS[@]}
N_JOBS=$((N_CELLS * N_SEEDS))
if [[ "${1:-plan}" == "plan" ]]; then
  echo "== SIGNAL pod sweep PLAN =="
  echo "cells: $N_CELLS   seeds: $N_SEEDS (start $SEED_START)   training jobs: $N_JOBS"
  echo "rhos needing R9 divisors: $RHOS_NEEDED"
  echo "workers: $WORKERS   episodes/job: $EPISODES"
  printf '%s\n' "${CELLS[@]}" | sed 's/^/  cell /'
  for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b <<<"$cell"; check_supported "$f" "$r" "$c" "$t"; done
  echo "capability check: all declared cells supported by this build."
  grep '^#UNSUPPORTED' "$0" | sed 's/^#/  /'
  exit 0
fi
[[ "${1:-}" == "run" ]] || { echo "usage: $0 plan|run"; exit 1; }
mkdir -p runs/logs

echo "== stage 0: gauntlet =="
$PY tests/test_vendor_env.py >runs/logs/gauntlet_vendor.log 2>&1
$PY tests/test_adapter.py    >runs/logs/gauntlet_adapter.log 2>&1
$PY tests/test_all.py        >runs/logs/gauntlet_all.log 2>&1
$PY -m signal_lab.hypotheses --self-test
echo "   all suites green."

echo "== stage 1: R9 divisors =="
$PY scripts/fit_scales.py --rhos "$RHOS_NEEDED"
[[ -n "$NEED_DP" ]] && $PY scripts/fit_scales.py --rhos "" --dr-poisson "$DR_LO" "$DR_HI"

echo "== stage 2: baselines =="
for r in ${RHOS_NEEDED//,/ }; do
  [[ -f "runs/baselines_rho${r}.json" ]] || $PY -m signal_lab.baselines --rho "$r"
done
[[ -n "$NEED_DP" && ! -f "runs/baselines_dp_${DR_LO}-${DR_HI}.json" ]] && \
  $PY -m signal_lab.baselines --demand-family dr_poisson

echo "== stage 3: training ($N_JOBS jobs, $WORKERS workers) =="
JOBFILE=$(mktemp)
for cell in "${CELLS[@]}"; do
  IFS='|' read -r f r c t b cl <<<"$cell"; check_supported "$f" "$r" "$c" "$t"
  ms=$(scale_for "$f" "$r" "$c")
  EXTRA=""
  [[ "$cl" != "-" ]] && EXTRA="obs_order_clip=$cl"
  [[ "$f" == "dr_poisson" ]] && EXTRA="$EXTRA dr_lambda_lo=$DR_LO dr_lambda_hi=$DR_HI"
  for ((k=0; k<N_SEEDS; k++)); do
    s=$((SEED_START + k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl")
    if [[ -f "runs/$tag/ckpt_best.pt" ]] && grep -q "\[signal\] done\." "runs/logs/$tag.log" 2>/dev/null; then
      continue; fi
    echo "$PY -m signal_lab.train --set tag=$tag content=$c seed=$s rho=$r beta=$b" \
         "topology=$t msg_scale=$ms total_episodes=$EPISODES demand_family=$f $EXTRA" \
         "> runs/logs/$tag.log 2>&1" >>"$JOBFILE"
  done
done
echo "   pending: $(wc -l <"$JOBFILE") of $N_JOBS"
xargs -a "$JOBFILE" -d'\n' -P "$WORKERS" -I{} bash -c '{}'
rm -f "$JOBFILE"
FAILED=0
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl <<<"$cell"
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl")
    grep -q "\[signal\] done\." "runs/logs/$tag.log" 2>/dev/null || { echo "   FAILED: $tag"; FAILED=1; }
  done; done
[[ $FAILED -eq 0 ]] || { echo "FAIL-CLOSED: training failures above -- fix, re-run (idempotent)."; exit 3; }

echo "== stage 4: evaluation =="
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl <<<"$cell"
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl")
    [[ -f "runs/$tag/eval/seed10000_rho${r}.json" ]] || \
      $PY -m signal_lab.evaluate --ckpt "runs/$tag/ckpt_best.pt" --episodes 50 --rho "$r" >>"runs/logs/$tag.log" 2>&1
    if [[ "$c" != "nocomm" ]]; then for iv in zeroed shuffled; do
      [[ -f "runs/$tag/eval/seed10000_rho${r}_${iv}.json" ]] || \
        $PY -m signal_lab.evaluate --ckpt "runs/$tag/ckpt_best.pt" --episodes 50 --rho "$r" --intervention "$iv" >>"runs/logs/$tag.log" 2>&1
    done; fi
  done; done
echo "   evaluation complete."

echo "== stage 5: analysis per (family,rho,beta) =="
# analysis groups: (family, rho, beta, clip) -- clip arms pair within their own level
for key in $(printf '%s\n' "${CELLS[@]}" | awk -F'|' '{print $1"|"$2"|"$5"|"$6}' | sort -u); do
  IFS='|' read -r f r b klip <<<"$key"
  noc=""; arms=""
  for cell in "${CELLS[@]}"; do IFS='|' read -r f2 r2 c t b2 cl <<<"$cell"
    [[ "$f2|$r2|$b2|$cl" == "$f|$r|$b|$klip" ]] || continue
    for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl")
      if [[ "$c" == "nocomm" ]]; then noc+="$tag,"; else arms+="$tag,"; fi
    done; done
  [[ -n "$arms" ]] || continue
  $PY -m signal_lab.stats --nocomm "${noc%,}" --arms "${arms%,}" --rho "$r" \
      | tee "runs/stats_${f}_rho${r}_b${b//./}.txt"
done

echo "== stage 6: hypotheses =="
$PY -m signal_lab.hypotheses --self-test
$PY -m signal_lab.hypotheses --stats "runs/stats_rho*.json" || true

echo "== stage 7: manifest =="
{ date -u +"%Y-%m-%dT%H:%M:%SZ"
  git -C . rev-parse HEAD 2>/dev/null || echo "no-git"
  sha256sum conf/signal.yaml signal_lab/*.py env/beer_game.py vendor/envs/beer_game_env.py \
            vendor/scripts/demand_families.py scripts/fit_scales.py runs/msg_scales.json
  echo "cells=$N_CELLS seeds=$N_SEEDS jobs=$N_JOBS episodes=$EPISODES"
} > runs/CAMPAIGN_MANIFEST.txt
echo "DONE. Manifest at runs/CAMPAIGN_MANIFEST.txt"
