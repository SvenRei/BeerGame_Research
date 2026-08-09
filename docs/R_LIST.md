# SIGNAL v6 — Registered Decision Ledger (R-List)
**Status:** amendment input. Every decision below deviates from, or refines, the v1.x/v2.0
registered design. Each entry states what it replaces, the evidence, and — critically —
its **provenance class**: `PRE` (decided from principle before seeing outcomes),
`DIAG` (adopted after a diagnosed instrument failure, before any confirmatory data),
`POST` (chosen after observing dev-seed outcomes). All dev evidence uses seeds 60–62
and the fixed evaluation space (base 10000); **no confirmatory seed has been touched.**

| id | decision | replaces | provenance | evidence |
|---|---|---|---|---|
| R1 | Action = order-up-to S over 41 bins on **[0,100]** | [0,160]×41 (QMIX-shared grid) | PRE | cold-start flood argument; ceiling verified non-binding (0% of eval actions at S=100 across all arms) |
| R2 | Plain discounted **MC returns**, adv = G − V, standardized | GAE(λ), vtarget = adv+V | DIAG | v5 critic collapse: logged EV ≡ 0.0 by construction; MC targets are collapse-immune |
| R3 | PPO clip **0.1**, grad-norm **0.2** | 0.2 / 0.5–10 | PRE | reference-implementation values; clip falsified as a lever in v5 round c4 |
| R4 | **Fixed budget, no early stopping**; budget milestones double as the V(budget) axis | patience + floor | DIAG | three v5 stopping incidents; milestones {2k, 6k, 12k, 24k} |
| R5 | **Single shared actor** + role one-hot | 4 separate actors | PRE | reference design; per-role behavior verified (echelon-specific S levels learned) |
| R6 | Critic **ReLU / 256**, N outputs | Tanh / 64 | DIAG | Tanh saturation hypothesis on the v5 collapse; honest-EV 0.6–0.9 across all 18 v6 runs |
| R7 | Selection = **trailing-3-mean of the held-out ρ=0.9 monitor** (seeds 60000+, disjoint from eval 10000+); low-ρ gate demoted to diagnostic | best single low-ρ gate reading | POST | D-A2 four-checkpoint audit: monitor ranking = eval ranking (Spearman 1.0) vs gate 0.2; smoothing counters winner's curse over ~60 readings |
| R8 | **S-grid ceiling retained at 100** (considered raise to 150; **rejected**) | — | POST (refuted) | ceiling share measured 0.000 at eval across arms; hypothesis withdrawn, grid unchanged |
| R9 | **Message input standardization**: each content divided by its stationary sd under the training distribution, measured by the registered protocol in `scripts/fit_scales.py` (random-order rollouts, seeds 90000+, before any training). Learned channel: divisor 100 (architectural range of the tanh head). | fixed ÷100 for all contents | DIAG→PRE | at ÷100 a demand message entered with sd 0.063 (16× weaker than the role one-hot); learned slope 0.3 vs optimum 4.0; at unit variance slopes reached 3.0–4.4 and V(raw) moved from ≈0 to +877. The **rule** (divide by measured sd) is registered; the incident that revealed it is disclosed. Dev runs used 6.0/6.1 vs measured 6.7 — deviation disclosed, runs retained. |
| R10 | Environment = **vendored, byte-unmodified** `BeerGame_Comm` env + families behind a physics-free numpy adapter; the user's own 89-test suite runs verbatim in CI | greenfield reimplementation | DIAG | reimplementation was found to omit the MIT/Sterman pipeline priming (initial on-order 16/16/16/12); trajectories now match the validated env to 0.000000000 over 50 steps; independent baseline refits agree (3870.6/2639.6) |
| R11 | Training recipe: **batch_episodes 8, lr_step 6000, budget 24000, anneal 24000 (absolute), gate 15 eps/ρ every 400** | batch 1, lr_step 2000, budget 8000, gate 5 eps/ρ per 200 | DIAG | batch-1 grad norms 20–150 vs a 0.2 clip → every update renormalized, entropy never fell (D-A1); lr_step 2000 froze the argmax policy at the third halving; gate SE ≈1300 at 5 eps could not rank checkpoints |
| R12 | **Selection/diagnostic file hygiene**: eval dumps are namespaced by checkpoint and by seed base; `report` reads only the canonical seed-base file | glob-and-merge discovery | DIAG | a monitor-space diagnostic dump silently overwrote 15/50 canonical episodes of one arm (2947.1 vs true 2925.5, paired se inflated 93.5→201.7); mechanism reproduced, fixed, regression-tested |

## Deviations from the v2.0 skeleton requiring explicit amendment text
1. **Budget milestones** {2k,6k,12k,24k} vs registered {1k,2k,4k,8k} — substitution-curve x-axis changes; slopes remain per-log₂ comparable.
2. **Selection machinery** (R7) differs from the skeleton's "held-out CRN gate, best-checkpoint, patience" — disclose with the Spearman evidence; identical machinery across all arms and (if run) both learners remains true.
3. **Content aliases**: v6 `arpred` ≡ skeleton `ar1_condmean`; v6 `dhatc` is the *frozen certified* forecaster (message content), replacing the v1.x trainable d̂ head (**the head is deleted** — forecasting is message content, not agent anatomy). Alias table to be frozen in the registration.
4. **msg_scale provenance** (R9): the constant was found after observing a dev failure; the registered object is the measurement *rule*, applied outcome-blind to all future cells (incl. per-ρ and per-regime divisors from `fit_scales.py`).
5. **Statistical stack**: BCa bootstrap CIs (scipy) in the dev tooling vs the registered studentized bootstrap-t; the confirmatory analyzer must implement bootstrap-t per §7 or the registration amended to BCa. **Open item — decide before hashing.**
6. **QMIX second learner**: not ported to v6. Options: (a) run the concordance arm on the certified legacy harness; (b) descope with disclosure. **Open item.**

## Dev-ladder evidence summary (seeds 60–62, eval base 10000, Holm within family)
V vs matched nocomm (seed-level mean ± between-seed se): arpred **+1029.8 ± 30.3**;
dhatc +928.4 ± 92.9; raw +876.8 ± 32.9; learned +761.9 ± 37.9; ip +176.4 ± 80.6 (n.s.).
All families sign-concordant. nocomm ≈ StaticBS in 3/3 seeds (paired |V| ≤ 238, all n.s.).
do(m): shuffled-listening positive and concordant in all five families; zeroed−shuffled
fragility gap ≈ +7.5k/+7.0k/+4.8k (engineered) vs +0.5k (learned) vs ≈0 (ip).
Emergent-protocol sign conventions arbitrary across seeds (r = +0.65, −0.54, +0.62).
