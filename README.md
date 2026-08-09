# SIGNAL v6 — value of demand-information sharing in the Beer Game

A minimal multi-agent PPO learner for a 4-echelon Beer Game, built to measure
**V = C(nocomm) − C(comm)** across a ladder of message contents. The learner is
deliberately boring; the message channel is the experiment.

**One design principle:** one architecture, all arms. The treatment is the *content* of a
fixed-width message slot — never the architecture around it. `nocomm` is the same agent
with the channel zeroed, so arms are byte-identical in parameters, initialization, and
gradient paths, and V is attributable to information content by construction.
Full rationale and registered decisions R1–R6: [`docs/DESIGN.md`](docs/DESIGN.md).

## Layout

```
vendor/                 THE ENVIRONMENT, vendored UNMODIFIED from BeerGame_Comm:
                          envs/beer_game_env.py, scripts/demand_families.py, conf/config.yaml
env/beer_game.py        ADAPTER ONLY -- no physics; numpy interface over vendor/
signal_lab/messages.py  MessageProvider: content ladder, topology routing, do(m) wrappers
signal_lab/agent.py     shared actor (GRU, role one-hot, S-grid head, msg head) + critic
signal_lab/train.py     rollout, PPO update, gates, checkpoints, CSVs
signal_lab/evaluate.py  deterministic eval + interventions -> eval/*.json
signal_lab/report.py    fail-closed verdicts vs in-project baselines
signal_lab/curves.py    fig14 (training) + fig15 (6-panel diagnostics, honest-EV canary)
signal_lab/baselines.py StaticBS + AR-conditional CondBS, refit against THIS env
sweep.py                contents x seeds -> jobs, --dry-run, trainlogs
conf/signal.yaml        the one config; every knob explicit; unknown --set keys rejected
scripts/fit_forecaster.py  (re)fits the frozen AR(1) forecaster for content=dhatc
tests/test_all.py       the invariant battery (below)
```

The package is `signal_lab`, not `signal` — `signal` collides with the Python stdlib.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/test_vendor_env.py                     # the env's own 89-test suite, verbatim
python tests/test_adapter.py                        # adapter faithfully exposes the env
python tests/test_all.py                            # project invariants
python -m signal_lab.baselines --rho 0.9            # fit the bars (report refuses without)
python -m signal_lab.train --set content=nocomm seed=60
python -m signal_lab.evaluate --ckpt runs/nocomm_s60/ckpt_best.pt --episodes 50
python -m signal_lab.report --arms nocomm_s60 --rho 0.9
python -m signal_lab.curves --arms nocomm_s60
python -m signal_lab.stats --arms nocomm_s60 --rho 0.9
```

## Statistics (`signal_lab/stats.py`)

Every inferential number is delegated to scipy/statsmodels (versions recorded in the
output JSON); numpy computes descriptives only. Per arm: mean/se/CVaR cost, per-echelon
bullwhip ratios, holding-vs-backorder decomposition, retailer ready rate, and channel
signaling (Pearson/Spearman vs d_prev). Paired vs the nocomm arm on identical CRN seeds:
V per episode, Cohen's d_z, P(V>0), paired t, Wilcoxon, BCa bootstrap CI, Schuirmann
TOST (`--tost-margin`), and Holm correction across the arm family. With a zeroed do(m)
dump present, the causal listening contrast (cost delta + action divergence) is added.
Paired against the baselines on the same eval draws: V vs StaticBS and vs CondBS with
paired t, Wilcoxon, BCa CI and TOST, plus gap-recovered with a BCa interval (requires
`baselines.py` schema 2 -- re-run it once). Across training-seed replicates (tags
`<arm>_s<NN>`): seed-level mean V with a BETWEEN-SEED se, one-sample t, BCa CI and sign
concordance -- because the eval seed space is shared, pooling episodes across replicates
would understate uncertainty. Fail-closed on missing dumps, seed mismatches, or NaNs.

Evaluate one training run completely:

```bash
python -m signal_lab.evaluate --ckpt runs/<tag>/ckpt_best.pt --episodes 50
python -m signal_lab.evaluate --ckpt runs/<tag>/ckpt_best.pt --episodes 50 --intervention zeroed
python -m signal_lab.stats --nocomm nocomm_s60 --arms <tag> --rho 0.9
```

Full ladder on dev seeds:

```bash
python sweep.py --contents nocomm,raw,ip,arpred,dhatc,learned --seeds 60,61,62 --dry-run
python sweep.py --contents nocomm,raw,ip,arpred,dhatc,learned --seeds 60,61,62
```

`--dry-run` prints every command with the full `--set` line — verify flag propagation
before spending compute. Trainlogs land in `runs/logs/`.

Everything is overridable inline, e.g.
`python -m signal_lab.train --set content=dhatc seed=61 rho=0.6 beta=0.5 total_episodes=16000`.
Unknown keys fail closed. Lists accept both `budget_milestones=[1000,2000]` and `1000,2000`.

`assets/forecaster_ar1r9.pt` (for `content=dhatc`) ships pre-fitted; regenerate with
`python scripts/fit_forecaster.py` if the demand family changes.

## Hypothesis → knob

| axis | knob |
|---|---|
| V(content) | `content` ∈ nocomm, raw, ip, arpred, dhatc, learned |
| V(ρ) | `rho` (AR(1) demand) |
| demand family | `demand_family` ∈ ar1, poisson (classic Beer Game mode, ρ ignored; `poisson_mu`) |
| V(geometry) | `topology` ∈ retailer_broadcast, neighbor |
| V(β) | `beta` (reward mix r_i = −(c_i + β·Σ_others)/100) |
| V(budget) | `budget_milestones` checkpoint snapshots |
| do(m) probes | `evaluate.py --intervention` ∈ honest, zeroed, shuffled, cross |

If an axis ever needs a code change instead of a config value, the design has failed.

## Run-artifact contract

Every run writes `runs/<tag>/` with: `config_resolved.yaml` (hash printed at start),
`command.txt`, `metrics_train.csv`, `metrics_gate.csv`, `metrics_update.csv`,
`ckpt_best.pt` / `ckpt_budget<N>.pt` / `ckpt_final.pt` (payloads carry config + seed;
tensors cloned), `eval/*.json`. `report.py` reads only this contract and is fail-closed:
missing baselines → NO-REF, missing run → NO-RUN, missing eval → NO-EVAL, exit ≠ 0 unless
all PASS.

## Seed spaces (disjoint by construction)

| space | base |
|---|---|
| training episodes | derived from `seed` |
| gate (selection, ρ ∈ {0.15, 0.45, 0.75}) | 50 000 |
| monitor (ρ = 0.9, logged, **never selects**) | 60 000 |
| eval / paired scoring | 10 000 |
| baseline fitting | 70 000 |

## The test battery (`tests/test_all.py`)

T-ENV determinism + cost accounting · T-ARPRED closed-form AR(1) · T-INTERV zeroed ≡
nocomm, shuffle preserves the multiset · T-FROZEN no gradient into the forecaster,
fail-closed loading · T-PARAM identical parameter counts and shapes across all contents ·
T-SYM `learned` agent with a zeroed provider is **bit-identical** to nocomm under CRN ·
T-GRAD the message head trains iff `content=learned` · T-SMOKE end-to-end run honors the
artifact contract and the report fail-closes correctly · T-SWEEP dry-run propagates every
override.

Green battery before any run is trusted; re-run after any edit.

## Diagnostics that watch the known failure modes

`fig15` plots the **honest explained variance** (1 − Var(G−V)/Var(G) against Monte-Carlo
returns) with a canary line at 0.05 — a collapsed critic is a red line on the first
figure, not a mystery seven campaigns later — plus |global state| (watching the fixed ÷100
divisor), entropy, value loss, grad norm, and approx-KL.

## Windows notes

Single-line commands only in `cmd.exe` (no `^` continuations). Activate with
`.venv\Scripts\activate`. Paths in `--ckpt` accept backslashes.
