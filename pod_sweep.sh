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
# Each line: family|rho|content|topology|beta|clip|bh   ("-" = default)
#   bh = backorder/holding ratio. "-" keeps the registered 0.5/1.0 (b/h = 2).
# LOW_RHO_EPISODES (default = EPISODES) extends the budget for rho < 0.9 cells only:
#   the campaign showed nocomm seed-CV of 0.17-0.34 there vs 0.018 at rho=0.9, i.e.
#   an unconverged control, which is what broke H2's registered grid. The extension
#   is applied IDENTICALLY to comm and nocomm arms so the comparison stays fair.
DR_LO="${DR_LO:-4}"; DR_HI="${DR_HI:-24}"
CELLS=()
# F_CONTENT ladder @ rho .9 (retailer_broadcast, beta 1)
for c in nocomm raw arpred dhatc ip learned; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|-|-"); done
# P1 (DP side): regime uncertainty, forecast (arpred) vs raw vs nocomm.
# rho label -1 keeps DP files/stats DISJOINT from the AR rho=0 group -- otherwise the
# analysis stage would pair DP arms against AR baselines on the shared label.
for c in nocomm raw arpred; do
  CELLS+=("dr_poisson|-1|$c|retailer_broadcast|1.0|-|-"); done
# P2 (garbling): clip levels chosen by the REGISTERED pre-flight audit
# (scripts/audit_garbling.py): c=6 destroys 62% of linearly-recoverable demand info
# from the order stream, c=8 destroys 47%. The originally planned c in {12,20} was
# measured at 11% and -14% -- too weak to test the Blackwell mechanism, so a null
# there would have been uninformative. Blackwell nesting min(o,6)=min(min(o,8),6)
# still holds, so the dose ordering Gamma(6) >= Gamma(8) >= 0 is well posed.
for cl in 6 8; do for c in nocomm raw; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|$cl|-"); done; done
# F_GEOMETRY on raw @ rho .9. retailer_broadcast is the direct-source arm (reused from
# F_CONTENT). upstream_only is the RELAYED arm -- H-SOURCE contrasts the two. The rest
# are placebos: downstream_only echoes each agent's own past order back at it, and
# no_neighbor wires nobody to anybody (bit-identical to nocomm; V must be EXACTLY 0).
# all_to_all is the information CEILING for the channel -- every stage hears every
# other -- and the benchmark retailer_broadcast must be measured against. It was absent
# from the original geometry family, which registered "who hears the retailer" rather
# than "how much connectivity"; without it the study has an upper bound it never tested.
# Geometry family, complete. Beyond direction and source, these probe DENSITY
# (all_to_all, full, skip), MIXING (neighbor_bidir: does a harmful channel degrade a
# helpful one when both are live?), REACH (the two single-link probes: is the value at
# the cleanest link or the most distorted one?) and a wrong-partner control that is
# live rather than empty.
for t in upstream_only downstream_only manufacturer_broadcast no_neighbor all_to_all \
         neighbor_bidir wrong_partner skip full link_top_only link_bottom_only; do
  CELLS+=("ar1|0.9|raw|$t|1.0|-|-"); done
# CONTENT x TOPOLOGY interaction. Every geometry cell above uses raw demand, so the
# study can say broadcast beats relay FOR RAW but not whether the ranking of contents
# survives a degraded channel. These three complete a 4x2 factorial -- {raw, forecast,
# learned forecast, emergent} x {broadcast, relay} -- because the other five corners
# already exist. Registered question: a forecast is a smoothed statistic, so it may
# survive relaying better than a noisy raw observation; if the content gap widens under
# relay, "what you send" starts to matter once the channel degrades.
for c in arpred dhatc learned; do
  CELLS+=("ar1|0.9|$c|upstream_only|1.0|-|-"); done
# H-TIME: staleness gradient (raw itself is lag 0, reused from F_CONTENT)
for c in raw_lag1 raw_lag2; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|-|-"); done
# H2 rho-gradient on raw with matched nocomm (per-rho pairing partner) + dhatc contrast
for r in 0 0.3 0.6; do for c in nocomm raw dhatc; do
  CELLS+=("ar1|$r|$c|retailer_broadcast|1.0|-|-"); done; done
# F_INCENTIVE: dhatc-only vs matched-beta nocomm
for b in 0 0.5; do for c in nocomm dhatc; do
  CELLS+=("ar1|0.9|$c|retailer_broadcast|$b|-|-"); done; done
# --- still outside this build (plan-time fail-close names them):
#UNSUPPORTED: content eps,dhat_ip,true_lambda      (messages.py rungs)
#UNSUPPORTED: QMIX second learner                  (legacy harness or descope)

# F3 robustness cell: backorder-heavy regime (b/h = 4), raw + nocomm only.
for c in nocomm raw; do CELLS+=("ar1|0.9|$c|retailer_broadcast|1.0|-|4"); done

# ---------------------------------------------------------------- H2 RE-RUN BLOCK
# The rho-grid cells were selected by a monitor hardcoded to rho=0.9 (R7-FIX in
# signal_lab/train.py). Symptom: nocomm-vs-StaticBS degraded monotonically with
# |rho - 0.9| (-803 / -309 / +68 / -28) while seed CV rose to 0.34. Same recipe, same
# 24000 episodes -- ONLY the selector changes -- so this is a re-selection, not a
# budget deviation. Tagged _v2 so the original cells stay on disk for before/after.
# Enable with:  H2_RERUN=1 ./pod_sweep.sh run
# GEO_ONLY=1: the geometry + content-x-topology questions ONLY, plus the two reference
# cells they are measured against. Use when the previous campaign's checkpoints are not
# available on this machine: retraining nocomm and raw-broadcast here costs 30 jobs and
# makes every new comparison internally consistent, which is preferable to pairing new
# cells against arms trained on a different machine with different measured divisors.
if [[ "${GEO_ONLY:-0}" == "1" ]]; then
  CELLS=("ar1|0.9|nocomm|retailer_broadcast|1.0|-|-"
         "ar1|0.9|raw|retailer_broadcast|1.0|-|-")
  for t in upstream_only downstream_only manufacturer_broadcast no_neighbor all_to_all \
           neighbor_bidir wrong_partner skip full link_top_only link_bottom_only; do
    CELLS+=("ar1|0.9|raw|$t|1.0|-|-"); done
  for c in arpred dhatc learned; do
    CELLS+=("ar1|0.9|$c|upstream_only|1.0|-|-"); done
fi

if [[ "${H2_RERUN:-0}" == "1" ]]; then
  CELLS=()
  for r in 0 0.3 0.6; do for c in nocomm raw dhatc; do
    CELLS+=("ar1|$r|$c|retailer_broadcast|1.0|-|-"); done; done
  TAG_SUFFIX="_v2"
fi

SUPPORTED_CONTENT="nocomm raw arpred dhatc ip learned raw_lag1 raw_lag2"
SUPPORTED_TOPO="retailer_broadcast neighbor upstream_only downstream_only manufacturer_broadcast no_neighbor all_to_all neighbor_bidir wrong_partner skip full link_top_only link_bottom_only"
SUPPORTED_FAMILY="ar1 poisson dr_poisson"

tag_of() { local f=$1 r=$2 c=$3 t=$4 b=$5 s=$6 cl=$7 bh=${8:--}
  local base="C_${f}_r${r//./}_${c}_${t:0:4}_b${b//./}${TAG_SUFFIX:-}"
  if [[ "$cl" != "-" ]]; then base="${base}_cl${cl}"; fi
  if [[ "$bh" != "-" ]]; then base="${base}_bh${bh}"; fi
  echo "${base}_s${s}"; }

check_supported() { local f=$1 r=$2 c=$3 t=$4
  [[ "$f" == "dr_poisson" && "$c" == "dhatc" ]] && { echo "FAIL-CLOSED: dhatc is AR(1)-certified only"; exit 2; }
  [[ " $SUPPORTED_FAMILY " == *" $f "* ]] || { echo "FAIL-CLOSED: family '$f' not in this build"; exit 2; }
  [[ " $SUPPORTED_CONTENT " == *" $c "* ]] || { echo "FAIL-CLOSED: content '$c' not in this build"; exit 2; }
  [[ " $SUPPORTED_TOPO " == *" $t "* ]]   || { echo "FAIL-CLOSED: topology '$t' not in this build"; exit 2; }
}

scale_for() { # family rho content -> divisor from runs/msg_scales.json (fail-closed)
  $PY - "$1" "$2" "$3" "$DR_LO" "$DR_HI" <<'PYEOF'
import json, sys
fam, rho, content = sys.argv[1], sys.argv[2], sys.argv[3]
key = (f"{fam}|rho{float(rho):g}" if fam == "ar1"
       else f"dr_poisson|{sys.argv[4]}-{sys.argv[5]}" if fam == "dr_poisson"
       else f"{fam}|{rho}")
t = json.load(open("runs/msg_scales.json"))
if key not in t or content not in t[key]:
    sys.exit(f"FAIL-CLOSED: no registered divisor for ({key},{content}) -- run stage 1")
if t[key][content] is None:
    sys.exit(f"FAIL-CLOSED: content '{content}' is DEGENERATE (constant message) at "
             f"{key}; its divisor is undefined and the cell is informationless -- "
             f"remove it from CELLS")
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
# F3: the robustness cell needs its OWN bars -- costs change the optimal base stock.
for bh in $(printf '%s\n' "${CELLS[@]}" | awk -F'|' '$7!="-"{print $7}' | sort -u); do
  [[ -f "runs/baselines_rho0.9_bh${bh}.json" ]] || \
    $PY -m signal_lab.baselines --rho 0.9 --holding-cost 0.5 \
        --backorder-cost "$(python3 -c "print(0.5*$bh)")"
done
# DP cells are analysed under rho label -1, so their bars must land in
# baselines_rho-1.json -- the file stats.py opens for that group.
DP_RHO=$(printf '%s\n' "${CELLS[@]}" | awk -F'|' '$1=="dr_poisson"{print $2; exit}')
[[ -n "$NEED_DP" && ! -f "runs/baselines_rho${DP_RHO}.json" ]] && \
  $PY -m signal_lab.baselines --demand-family dr_poisson --rho "$DP_RHO"

echo "== stage 3: training ($N_JOBS jobs, $WORKERS workers) =="
JOBFILE=$(mktemp)
for cell in "${CELLS[@]}"; do
  IFS='|' read -r f r c t b cl bh <<<"$cell"; check_supported "$f" "$r" "$c" "$t"
  ms=$(scale_for "$f" "$r" "$c")
  EXTRA=""
  if [[ "$cl" != "-" ]]; then EXTRA="obs_order_clip=$cl"; fi
  if [[ "$f" == "dr_poisson" ]]; then EXTRA="$EXTRA dr_lambda_lo=$DR_LO dr_lambda_hi=$DR_HI"; fi
  # F3: b/h ratio -> hold h fixed at 0.5 and scale b, so the holding scale (and thus
  # the S-grid's meaning) is unchanged and only the asymmetry moves.
  if [[ "$bh" != "-" ]]; then EXTRA="$EXTRA holding_cost=0.5 backorder_cost=$(python3 -c "print(0.5*$bh)")"; fi
  # F1: unconverged control at low rho -> longer, EQUAL budget for both arms there
  EPS_CELL="$EPISODES"
  if [[ "$f" == "ar1" && "$r" != "0.9" && -n "${LOW_RHO_EPISODES:-}" ]]; then
    EPS_CELL="$LOW_RHO_EPISODES"
  fi
  for ((k=0; k<N_SEEDS; k++)); do
    s=$((SEED_START + k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
    if [[ -f "runs/$tag/ckpt_best.pt" ]] && grep -q "\[signal\] done\." "runs/logs/$tag.log" 2>/dev/null; then
      continue; fi
    echo "$PY -m signal_lab.train --set tag=$tag content=$c seed=$s rho=$r beta=$b" \
         "topology=$t msg_scale=$ms total_episodes=$EPS_CELL demand_family=$f $EXTRA" \
         "> runs/logs/$tag.log 2>&1" >>"$JOBFILE"
  done
done
echo "   pending: $(wc -l <"$JOBFILE") of $N_JOBS"
xargs -a "$JOBFILE" -d'\n' -P "$WORKERS" -I{} bash -c '{}'
rm -f "$JOBFILE"
FAILED=0
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl bh <<<"$cell"
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
    grep -q "\[signal\] done\." "runs/logs/$tag.log" 2>/dev/null || { echo "   FAILED: $tag"; FAILED=1; }
  done; done
[[ $FAILED -eq 0 ]] || { echo "FAIL-CLOSED: training failures above -- fix, re-run (idempotent)."; exit 3; }

echo "== stage 4: evaluation =="
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl bh <<<"$cell"
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
    [[ -f "runs/$tag/eval/seed10000_rho${r}.json" ]] || \
      $PY -m signal_lab.evaluate --ckpt "runs/$tag/ckpt_best.pt" --episodes 50 --rho "$r" >>"runs/logs/$tag.log" 2>&1
    if [[ "$c" != "nocomm" ]]; then for iv in zeroed shuffled; do
      [[ -f "runs/$tag/eval/seed10000_rho${r}_${iv}.json" ]] || \
        $PY -m signal_lab.evaluate --ckpt "runs/$tag/ckpt_best.pt" --episodes 50 --rho "$r" --intervention "$iv" >>"runs/logs/$tag.log" 2>&1
    done; fi
  done; done
# H-BUDGET -- the substitution curve. train.py already SAVES ckpt_budget<N>.pt at each
# milestone (2k/6k/12k/24k) at no extra training cost, but nothing ever scored them, so
# the registered V(budget) axis produced no data. Registered read: information sharing
# and learning time are partial SUBSTITUTES, so V should DECLINE as the budget grows --
# a longer-trained no-communication policy closes part of the gap on its own. If V is
# flat in budget, information and computation are complements, not substitutes, at this
# scale. evaluate.py namespaces non-best checkpoints (__ckpt_budget<N>), so these dumps
# can never overwrite the headline arm result. Restricted to the F_CONTENT ladder.
BUDJOBS=$(mktemp)
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl bh <<<"$cell"
  [[ "$f" == "ar1" && "$r" == "0.9" && "$cl" == "-" && "$bh" == "-" && "$b" == "1.0" \
     && "$t" == "retailer_broadcast" ]] || continue
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
    for m in 2000 6000 12000; do          # 24000 == ckpt_best's budget, already scored
      [[ -f "runs/$tag/ckpt_budget${m}.pt" ]] || continue
      [[ -f "runs/$tag/eval/seed10000_rho${r}__ckpt_budget${m}.json" ]] || \
        echo "$PY -m signal_lab.evaluate --ckpt runs/$tag/ckpt_budget${m}.pt --episodes 50 \
             --rho $r >> runs/logs/$tag.log 2>&1" >>"$BUDJOBS"
    done
  done
done
if [[ -s "$BUDJOBS" ]]; then
  echo "   $(wc -l <"$BUDJOBS") budget-milestone evaluations (H-BUDGET), $WORKERS workers"
  xargs -a "$BUDJOBS" -d'\n' -P "$WORKERS" -I{} bash -c '{}'
fi
rm -f "$BUDJOBS"

# P2 DIRECT TEST -- do(obs). Scramble the OBSERVED incoming-order field of the three
# upstream stages on already-trained policies. P2's between-arm null infers that
# no-communication policies never mined the order stream for demand; this measures it.
# Pre-registered read: if cost(nocomm | obs_shuffled) - cost(nocomm | honest) is ~0,
# the policy provably ignores that history and Raghunathan's redundancy mechanism is
# ABSENT under learning. A large positive delta would refute the P2 interpretation and
# mean the clip simply failed to bind. Run on nocomm AND raw (raw is the contrast: it
# has a channel, so it should depend on the order stream even less).
DOBSJOBS=$(mktemp)
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl bh <<<"$cell"
  [[ "$f" == "ar1" && "$r" == "0.9" && "$cl" == "-" && "$bh" == "-" && "$b" == "1.0" \
     && "$t" == "retailer_broadcast" ]] || continue
  [[ "$c" == "nocomm" || "$c" == "raw" ]] || continue
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
    [[ -f "runs/$tag/eval/seed10000_rho${r}_obs_shuffled.json" ]] || \
      echo "$PY -m signal_lab.evaluate --ckpt runs/$tag/ckpt_best.pt --episodes 50 \
           --rho $r --obs-intervention obs_shuffled >> runs/logs/$tag.log 2>&1" >>"$DOBSJOBS"
  done
done
if [[ -s "$DOBSJOBS" ]]; then
  echo "   $(wc -l <"$DOBSJOBS") do(obs) evaluations (P2 direct test), $WORKERS workers"
  xargs -a "$DOBSJOBS" -d'\n' -P "$WORKERS" -I{} bash -c '{}'
fi
rm -f "$DOBSJOBS"
echo "   in-distribution evaluation complete."

echo "== stage 4b: ZERO-SHOT OOD transfer (ALL rho=.9 arms) =="
# Policies trained on ar1 are evaluated, WITHOUT retraining, on the vendored stress
# decks. Training on those decks would be confounded (fixed calendar, memorisable);
# transferring to them is not -- the schedule is unanticipated, so the retailer's
# observation is a genuine early warning. Labels keep dumps disjoint.
# 1020 forward-pass evaluations: queue them and run WORKERS-parallel like stage 3,
# otherwise this stage alone is ~2.3 h serial.
OODJOBS=$(mktemp)
for cell in "${CELLS[@]}"; do IFS='|' read -r f r c t b cl bh <<<"$cell"
  # H-SHOCK EXTENSION: transfer EVERY in-distribution rho=0.9 arm, not just nocomm+raw.
  # Costs no training -- these are forward passes on checkpoints the campaign already
  # produced -- and it turns H-SHOCK from a single-content, single-topology claim into
  # a full mechanism test:
  #   * contents  -> does the early-warning premium depend on WHAT is sent? (arpred is
  #     an AR-calibrated shrinkage rule and dhatc an AR-certified neural forecaster;
  #     both face inputs outside their calibration under a shock, so brittleness by
  #     message type becomes measurable.)
  #   * topologies -> H-SOURCE under shock. Relay costs 524 in distribution; each hop's
  #     delay compounds against a level change, so the penalty should be LARGER OOD.
  #   * lags       -> H-TIME under shock. A stale statistic is mildly worse; a stale
  #     WARNING is nearly worthless, so the endpoint gap should exceed its 226.
  #   * placebos   -> falsification: downstream_only / no_neighbor must stay flat OOD.
  #     If a self-echo acquires value under shock, the early-warning story is wrong.
  # Excluded by design: clip and b/h cells (different observation map / cost regime,
  # so their nocomm reference is not the one used here) and non-ar1 families.
  [[ "$f" == "ar1" && "$r" == "0.9" && "$cl" == "-" && "$bh" == "-" && "$b" == "1.0" ]] || continue
  for sc_pair in "black_swan:-3" "extreme_chaos:-4"; do
    sc="${sc_pair%%:*}"; lab="${sc_pair##*:}"
    for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
      [[ -f "runs/$tag/eval/seed10000_rho${lab}.json" ]] || \
        echo "$PY -m signal_lab.evaluate --ckpt runs/$tag/ckpt_best.pt --episodes 50 \
             --scenario $sc --rho $lab >> runs/logs/$tag.log 2>&1" >>"$OODJOBS"
      if [[ "$c" != "nocomm" ]]; then for iv in zeroed shuffled; do
        [[ -f "runs/$tag/eval/seed10000_rho${lab}_${iv}.json" ]] || \
          echo "$PY -m signal_lab.evaluate --ckpt runs/$tag/ckpt_best.pt --episodes 50 \
               --scenario $sc --rho $lab --intervention $iv >> runs/logs/$tag.log 2>&1" >>"$OODJOBS"
      done; fi
    done
  done
done
if [[ -s "$OODJOBS" ]]; then
  echo "   $(wc -l <"$OODJOBS") OOD evaluations, $WORKERS workers"
  xargs -a "$OODJOBS" -d'\n' -P "$WORKERS" -I{} bash -c '{}'
fi
rm -f "$OODJOBS"
# OOD reference bars: a base-stock policy fitted ON the shock (the "oracle who knew")
for sc_pair in "black_swan:-3" "extreme_chaos:-4"; do
  sc="${sc_pair%%:*}"; lab="${sc_pair##*:}"
  [[ -f "runs/baselines_rho${lab}.json" ]] || \
    $PY -m signal_lab.baselines --demand-family "$sc" --rho "$lab" || true
done
echo "   OOD transfer complete."

echo "== stage 5: analysis per (family,rho,beta) =="
# analysis groups: (family, rho, beta, clip) -- clip arms pair within their own level
for key in $(printf '%s\n' "${CELLS[@]}" | awk -F'|' '{print $1"|"$2"|"$5"|"$6"|"$7}' | sort -u); do
  IFS='|' read -r f r b klip kbh <<<"$key"
  noc=""; arms=""
  for cell in "${CELLS[@]}"; do IFS='|' read -r f2 r2 c t b2 cl bh <<<"$cell"
    [[ "$f2|$r2|$b2|$cl|$bh" == "$f|$r|$b|$klip|$kbh" ]] || continue
    for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k)); tag=$(tag_of "$f" "$r" "$c" "$t" "$b" "$s" "$cl" "$bh")
      if [[ "$c" == "nocomm" ]]; then noc+="$tag,"; else arms+="$tag,"; fi
    done; done
  [[ -n "$arms" ]] || continue
  # NOTE the if-form: `[[ ... ]] && X` inside an assignment returns 1 when the test
  # is false, and under `set -e` that kills the whole script -- silently, before the
  # first stats call. That exact failure shipped once; keep the explicit if.
  GLABEL="${f}_b${b//./}"
  if [[ "$klip" != "-" ]]; then GLABEL="${GLABEL}_cl${klip}"; fi
  if [[ "$kbh" != "-" ]]; then GLABEL="${GLABEL}_bh${kbh}"; fi
  $PY -m signal_lab.stats --nocomm "${noc%,}" --arms "${arms%,}" --rho "$r" \
      --tag "$GLABEL" | tee "runs/stats_${f}_rho${r}_${GLABEL}.txt"
done

echo "== stage 5b: OOD analysis =="
for lab in -3 -4; do
  noc=""; arms=""
  for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k))
    noc+="$(tag_of ar1 0.9 nocomm retailer_broadcast 1.0 "$s" - -),"
  done
  # every transferred arm, paired against the same matched nocomm references
  for cell in "${CELLS[@]}"; do IFS='|' read -r f2 r2 c2 t2 b2 cl2 bh2 <<<"$cell"
    [[ "$f2" == "ar1" && "$r2" == "0.9" && "$cl2" == "-" && "$bh2" == "-" && "$b2" == "1.0" ]] || continue
    [[ "$c2" == "nocomm" ]] && continue
    for ((k=0; k<N_SEEDS; k++)); do s=$((SEED_START+k))
      tg=$(tag_of "$f2" "$r2" "$c2" "$t2" "$b2" "$s" "$cl2" "$bh2")
      [[ -f "runs/$tg/eval/seed10000_rho${lab}.json" ]] && arms+="$tg,"
    done
  done
  [[ -n "$arms" ]] || continue
  $PY -m signal_lab.stats --nocomm "${noc%,}" --arms "${arms%,}" --rho "$lab" \
      --tag ood | tee "runs/stats_OOD_rho${lab}.txt"
done

echo "== stage 5c: consolidated results sheet =="
$PY -m signal_lab.collate

echo "== stage 6: hypotheses =="
$PY -m signal_lab.hypotheses --self-test
$PY -m signal_lab.hypotheses --stats "runs/stats_rho*.json" || true

echo "== stage 7: manifest =="
{ date -u +"%Y-%m-%dT%H:%M:%SZ"
  git -C . rev-parse HEAD 2>/dev/null || echo "no-git"
  sha256sum conf/signal.yaml signal_lab/*.py env/beer_game.py vendor/envs/beer_game_env.py \
            vendor/scripts/demand_families.py scripts/fit_scales.py runs/msg_scales.json \
            runs/RESULTS.csv 2>/dev/null
  echo "cells=$N_CELLS seeds=$N_SEEDS jobs=$N_JOBS episodes=$EPISODES"
} > runs/CAMPAIGN_MANIFEST.txt
echo "DONE. Manifest at runs/CAMPAIGN_MANIFEST.txt"