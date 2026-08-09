#!/usr/bin/env bash
# pod_bootstrap.sh -- one command from a bare Linux pod to finished figures.
#
#   chmod +x pod_bootstrap.sh && ./pod_bootstrap.sh
#
# Stages: system deps -> venv -> python deps -> gauntlet -> campaign (pod_sweep.sh)
#         -> figures -> archive. Every stage is idempotent: re-running resumes.
# Everything streams to campaign.log as well as the console, so a dropped SSH
# session never loses the record (and nohup/tmux use is safe).
#
# Env knobs (all optional):
#   WORKERS      parallel training jobs      (default: nproc-2, min 1)
#   N_SEEDS      seeds per cell              (default 15)
#   SEED_START   first confirmatory seed     (default 30)
#   EPISODES     episodes per run            (default 24000)
#   SKIP_APT=1   skip apt-get (no sudo / image already has python3-venv)
#   PLAN_ONLY=1  print the job matrix and exit
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
LOG="$ROOT/campaign.log"
exec > >(tee -a "$LOG") 2>&1
echo "=============================================================="
echo " SIGNAL campaign bootstrap  |  $(date -u +%FT%TZ)  |  $ROOT"
echo "=============================================================="

WORKERS="${WORKERS:-$(( $(nproc) - 2 > 0 ? $(nproc) - 2 : 1 ))}"
N_SEEDS="${N_SEEDS:-15}"
SEED_START="${SEED_START:-30}"
EPISODES="${EPISODES:-24000}"
export WORKERS N_SEEDS SEED_START EPISODES

step() { echo; echo "----- $* -----"; }

# ---------------------------------------------------------------- 1 system deps
step "1/7 system dependencies"
if [[ "${SKIP_APT:-0}" == "1" ]]; then
  echo "skipped (SKIP_APT=1)"
elif command -v apt-get >/dev/null 2>&1; then
  SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
  $SUDO apt-get update -qq && \
  $SUDO apt-get install -y -qq python3 python3-venv python3-pip zip git >/dev/null
  echo "apt packages ready"
else
  echo "no apt-get; assuming python3/pip present"
fi
python3 --version

# ---------------------------------------------------------------- 2 venv
step "2/7 virtual environment"
# USE_SYSTEM_PYTHON=1 skips the venv (useful on slim images or where disk is tight;
# a torch wheel is ~2 GB, so check free space before creating one).
if [[ "${USE_SYSTEM_PYTHON:-0}" == "1" ]]; then
  echo "using system python (USE_SYSTEM_PYTHON=1)"
else
  AVAIL_GB=$(df -Pk . | awk 'NR==2{print int($4/1048576)}')
  if [[ "$AVAIL_GB" -lt 6 ]]; then
    echo "FAIL-CLOSED: only ${AVAIL_GB} GB free; a torch install needs ~5 GB."
    echo "  free space, or re-run with USE_SYSTEM_PYTHON=1 if deps are present."
    exit 1
  fi
  [[ -d .venv ]] || python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip -q
fi
echo "python: $(which python3)"

# ---------------------------------------------------------------- 3 python deps
step "3/7 python dependencies"
PIPFLAGS=""
if [[ "${USE_SYSTEM_PYTHON:-0}" == "1" ]]; then
  # PEP 668: distro pythons refuse installs without an explicit override.
  python3 -m pip install -q --dry-run pip >/dev/null 2>&1 || PIPFLAGS="--break-system-packages"
fi
python3 -m pip install -q $PIPFLAGS -r requirements.txt 2>&1 | tail -2 || \
  { echo "FAIL-CLOSED: dependency install failed (see above)."; exit 1; }
python3 - <<'PYEOF'
import importlib
for m in ("torch","numpy","scipy","statsmodels","matplotlib","yaml",
          "pettingzoo","gymnasium"):
    mod = importlib.import_module(m)
    print(f"  {m:<12} {getattr(mod,'__version__','ok')}")
PYEOF

# ---------------------------------------------------------------- 4 gauntlet
step "4/7 verification gauntlet (nothing runs unless these pass)"
python3 tests/test_vendor_env.py 2>&1 | tail -3
python3 tests/test_adapter.py    2>&1 | tail -3
python3 tests/test_all.py        2>&1 | tail -2
python3 -m signal_lab.hypotheses --self-test
echo "gauntlet: all green"

# ---------------------------------------------------------------- 5 campaign
step "5/7 campaign  (workers=$WORKERS seeds=$N_SEEDS from $SEED_START)"
chmod +x pod_sweep.sh
./pod_sweep.sh plan
if [[ "${PLAN_ONLY:-0}" == "1" ]]; then
  echo "PLAN_ONLY=1 -- stopping before training."; exit 0
fi
echo "starting at $(date -u +%FT%TZ). Resume-safe: re-run this script if interrupted."
./pod_sweep.sh run

# ---------------------------------------------------------------- 6 figures
step "6/7 figures"
mkdir -p figs
ARMS=$(ls -d runs/C_* 2>/dev/null | xargs -n1 basename 2>/dev/null | paste -sd, - || true)
if [[ -n "$ARMS" ]]; then
  python3 -m signal_lab.curves --arms "$ARMS" 2>&1 | tail -3
fi
# ladder summary figure: seed-level V by content family with between-seed error bars
python3 - <<'PYEOF'
import glob, json, os, re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RE = re.compile(r"^(.*)_s(\d+)$")
fam = {}
for p in glob.glob("runs/stats_*.json"):
    d = json.load(open(p))
    for arm, pb in d.get("paired_vs_nocomm", {}).items():
        m = RE.match(arm)
        if m: fam.setdefault(m.group(1), []).append(pb["V_mean"])
if not fam:
    print("  no paired blocks yet -- ladder figure skipped"); raise SystemExit
names = sorted(fam, key=lambda k: -np.mean(fam[k]))
mu = [np.mean(fam[k]) for k in names]
se = [np.std(fam[k], ddof=1)/np.sqrt(len(fam[k])) if len(fam[k]) > 1 else 0 for k in names]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(names)), mu, yerr=se, capsize=5, color="#4C72B0")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("C_ar1_", "").replace("_reta", "") for n in names],
                   rotation=30, ha="right", fontsize=8)
ax.set_ylabel("V vs matched nocomm (cost units)")
ax.set_title("Value of information sharing by message content\n"
             "(seed-level mean, between-seed SE)")
for i, k in enumerate(names):
    ax.text(i, mu[i], f"n={len(fam[k])}", ha="center", va="bottom", fontsize=7)
fig.tight_layout(); fig.savefig("figs/fig20_ladder_V.png", dpi=150)
print(f"  wrote figs/fig20_ladder_V.png  ({len(names)} families)")
PYEOF

# ---------------------------------------------------------------- 7 archive
step "7/7 archive"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar -czf "campaign_${STAMP}.tar.gz" runs/stats_*.json runs/stats_*.txt \
    runs/CAMPAIGN_MANIFEST.txt runs/msg_scales.json runs/baselines_*.json \
    figs docs 2>/dev/null || true
sha256sum "campaign_${STAMP}.tar.gz" | tee -a runs/CAMPAIGN_MANIFEST.txt
echo
echo "=============================================================="
echo " COMPLETE  |  $(date -u +%FT%TZ)"
echo "   results  : runs/stats_*.txt , runs/stats_rho*.json"
echo "   figures  : figs/"
echo "   manifest : runs/CAMPAIGN_MANIFEST.txt"
echo "   archive  : campaign_${STAMP}.tar.gz"
echo "   full log : campaign.log"
echo "=============================================================="
