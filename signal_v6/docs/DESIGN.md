# SIGNAL v6 — Agent Design Concept & Implementation Guide

**Purpose:** one minimal agent that learns the Beer Game reliably and carries the *entire*
hypothesis ladder — V(content), V(ρ), V(geometry), V(budget), V(β) — through configuration
alone. Greenfield everywhere except the message machinery, which is the treatment and the
point of the study.

**Status of this document:** design concept + implementation guide, no code. Every choice
below is either (a) inherited from the reference implementation that demonstrably learns
this game (`SvenRei/BeerGame`), (b) forced by a hypothesis, or (c) a registered decision
flagged for the amendment. Nothing is aesthetic.

---

## 1. The one design principle everything follows from

> **One architecture, all arms. The treatment is the *content* of the message channel,
> never the architecture around it.**

Every agent, in every arm of every axis, is the *same network with the same parameter
count, the same initialization under the same seed, and the same gradient paths* — except
for what flows through a fixed-width message slot. The nocomm arm is not a different
agent; it is the same agent with the channel **zeroed**.

Why this is the load-bearing decision:

1. **V is attributable by construction.** V = C(nocomm) − C(comm) currently rests on the
   claim that arms differ only in information. In v5 that claim is enforced by discipline
   (byte-identical comm machinery, revert-validated gates). In v6 it is enforced by
   *structure*: there is nothing else that can differ.
2. **It dissolves an entire bug class.** The A18/A19 d̂-head continuity issue, the
   `separate_frozen` aux-gate gap, and the d̂-head-active-in-nocomm finding from the D1
   dry-run were all instances of one failure mode: *architecture varying across arms*.
   Zero-channel nocomm makes that failure mode unrepresentable.
3. **CRN pairing becomes exact.** Identical parameter shapes and identical RNG consumption
   order across arms means common-random-number pairing holds at the trajectory level, and
   the CRN determinism tripwire ("identical gate series ⇒ dead flag") becomes a designed
   *test* instead of an accidental diagnostic.

Corollary: **forecasting is message content, not agent anatomy.** The v5 d̂ head is gone.
If a forecast exists (dhatc), it is produced *outside* the actor by the frozen certified
forecaster and injected through the channel like any other content. An actor never grows
an organ for one arm.

---

## 2. Module map

Nine files. Each has one responsibility; consumers touch only the run-artifact contract
(§8), never each other's internals.

| module | responsibility | provenance |
|---|---|---|
| `env/beer_game.py` | physics + demand families + global state | **keep** (validated; unchanged) |
| `signal/messages.py` | MessageProvider: content ladder, topology routing, interventions | **the preserved treatment** (port semantics from v5) |
| `signal/agent.py` | shared actor, critic, learned-message head | greenfield |
| `signal/train.py` | rollout, update, gate, checkpoints, CSV | greenfield |
| `signal/evaluate.py` | deterministic eval + do(m) probes → run-dir JSON | greenfield core, probes ported later |
| `signal/report.py` | verdicts, V, ladder table — fail-closed | port the gate-fix design |
| `signal/curves.py` | fig14/fig15 + critic-health panel | port + one new panel |
| `sweep.py` | axes × seeds → jobs; dry-run; trainlogs | port runner design (incl. `--dry-run`) |
| `conf/signal.yaml` | one flat config; every knob explicit | greenfield |

Kept outside, unmodified: `scripts/baselines.py` (the yardstick — rewriting it moves the
goalposts), the certified `forecaster_ar1r9.pt`, the seed-space definitions, the reference
numbers, and (later, at amendment time) the prereg/manifest/confirmatory stack from
`legacy/`.

---

## 3. The message system (`messages.py`) — the preserved core

### 3.1 Interface

A `MessageProvider` is a pure mapping, called once per environment step:

```
(env_state, observations, t, rng)  →  m ∈ R^{N × M}
```

with **M fixed across the entire study** (M = width of the widest content; narrower
contents zero-pad). Fixed M is what makes "one architecture, all arms" true: the actor's
input dimension never changes across the ladder.

### 3.2 The content ladder

| content | produces | trainable? | notes |
|---|---|---|---|
| `nocomm` | zeros | no | the control; also the zero-intervention |
| `raw` | retailer's observed demand d_{t−1}, routed by topology | no | pure function of env state |
| `ip` | sender's inventory position | no | pure function of env state |
| `arpred` | analytic AR(1) conditional mean E[d_t \| d_{t−1}; ρ, μ] | no | closed form; unit-tested against the formula |
| `dhatc` | frozen certified forecaster output | **no** (frozen) | loaded ckpt; `requires_grad=False`; no aux loss exists in v6 to mis-gate |
| `learned` | actor's message head output | **yes** — the only trainable content | DIAL-style differentiable channel: in the update's in-graph recompute the sender->receiver path stays **alive** -- receivers' task loss is exactly what trains the sender's message head (there is no aux loss in v6, so the v5 aux-detach fix is obsolete by construction) |

Everything except `learned` lives entirely in `messages.py` as stateless functions of the
environment — the agent cannot even see how they're made. `learned` is the single
exception: the provider routes the actor's message head output. This shrinks the agent to
its minimum and makes each content independently unit-testable without touching training.

### 3.3 Topology = a routing matrix

Geometry (V(geometry)) is an adjacency/routing matrix applied by the provider:
`retailer_broadcast`, `neighbor`, and any future topology are config values, zero code in
the agent. The agent receives "a vector arrived"; it never knows from where.

### 3.4 Interventions are provider wrappers

do(m) probes — honest / shuffled / cross / zeroed / identity-replay — wrap a provider at
evaluation time. Because the channel is the only treatment surface, interventions are
complete by construction: there is no side path a message can take around the wrapper.
(`zeroed` is literally the nocomm provider — one implementation, two names, which is
itself a nice consistency check.)

---

## 4. The agent (`agent.py`)

### 4.1 Actor — one network, shared across all four echelons

```
input   [ obs/100 ‖ role_onehot(4) ‖ m/100 ]
trunk   Linear → ReLU → Linear → ReLU        (width 256)
belief  GRU(256)                              (POMDP memory)
heads   action:  Linear → categorical over S-grid
        message: Linear → M                   (used only when content=learned)
init    orthogonal, gain 0.01 on heads        (near-uniform cold start)
```

Design notes, each traceable to evidence:

- **Parameter sharing + role one-hot** replaces v5's four separate actors. This is the
  reference's design, a ~4× sample-efficiency difference, and one of the enumerated
  untested deltas. Role heterogeneity survives through the one-hot; §9 includes a
  per-role behavior check in case it doesn't.
- **ReLU trunk, width 256** — the reference's, scale-tolerant. Not a single `Tanh`
  anywhere in v6 (pending the probe's verdict on the D1 critic, this is either the fix or
  merely harmless).
- **The message head always exists but is inert outside `learned`.** Constructing it
  unconditionally keeps parameter counts identical across arms; a test asserts it receives
  zero gradient in every other content (the D1 dry-run question, answered structurally).
- **Action = order-up-to level S over a categorical grid** — kept, because base-stock
  semantics is what makes AR_CondBS/AR_StaticBS the right comparators and what the thesis
  argues about. **Registered decision R1:** grid tightened from [0,160]×41 to
  **[0,100]×41**. A near-uniform cold start over [0,160] orders up to E[S]≈80 — double
  anything sensible — floods the chain, explodes backlogs, and (per the D1 probe
  hypothesis) drives the state magnitudes that killed the critic. [0,100] halves the
  cold-start flood and still contains every base-stock optimum with room. Cost: breaks
  bin-identity with QMIX's G2 grid; acceptable because QMIX is already adjudicated
  UNADJUDICABLE, and noted for the amendment.

### 4.2 Critic — centralized, boring, alive

```
input   global_state / 100
trunk   Linear → ReLU → Linear → ReLU        (width 256)
output  N values (one per agent)
```

- **ReLU** (reference), **fixed ÷100** (reference). No running normalizers: they make
  checkpoints stateful and eval-parity fragile, and the reference proves fixed-divisor +
  ReLU suffices. If state magnitudes stay bounded (tighter grid, working learner), the
  divisor is adequate; the health canary (§7) watches this assumption.
- **N outputs, β in the reward.** The β axis (cost sharing) makes per-agent rewards differ
  for β<1, so per-agent values are needed there. At β=1 the N outputs are redundant — and
  that redundancy is the *price of one architecture across the whole β axis*, which is the
  meta-principle. Axis symmetry beats micro-simplicity.

### 4.3 What the agent does **not** have

No d̂ head. No auxiliary losses. No forecast anatomy. No per-arm construction branches.
The full trainable surface is: trunk, GRU, action head, message head, critic. Five pieces.

---

## 5. Learning (`train.py`)

Reference recipe, adopted wholesale unless a hypothesis forbids it:

| element | v6 value | rationale |
|---|---|---|
| reward | r_i = −(c_i + β·Σ_{j≠i} c_j) / 100 | β is the incentive axis; ÷100 keeps value targets O(10) |
| returns | plain discounted Monte-Carlo | unbiased, no bootstrap, **immune to critic collapse** — with the ledger's history, robustness beats GAE's variance reduction. Registered decision R2 |
| advantage | G − V, standardized per update | reference |
| updates | every episode (or batch=2), k_epochs 4 | c5-validated step-count model |
| optimizer | Adam; actor 3e-4, critic 1e-3, separate | reference / v3.4 parity |
| **lr schedule** | **StepLR ×0.5 every 2000 episodes** | the reference's missing stabilizer — the sharpening-without-decay blowups (c6 s61, c7 s60/s62 post-bottom) are exactly what this prevents |
| entropy | 0.02 → 0, annealed over an **absolute** episode count | decoupled from `total_episodes` — the D1/c7 coupling (anneal length silently scaling with budget) violated one-delta-per-run and is banned by construction |
| clip / grad-norm | 0.1 / 0.2 | reference values; registered decision R3 (v5 used 0.2/0.5–10) |
| warm-up | 1000 episodes | reference |
| **stopping** | **none — fixed budget, no patience** | early stopping produced three incidents (c6 s61 starvation, c7 s61 pre-anneal death, the stall/footer confusion) to save compute the campaign doesn't need to save. Budget milestones double as the V(budget) axis. Registered decision R4 |
| selection | best gate checkpoint | unchanged protocol: gate ρ∈{.15,.45,.75}, never select on 0.9; add a **monitor-only** ρ=0.9 dev trace (plotted, never selecting) so the c7 gate/deployment decoupling is visible without leaking |

---

## 6. Hypothesis → knob map

The whole study is this table. If an axis ever needs a code change instead of a config
value, the design has failed and should be revisited.

| axis | knob | code path touched |
|---|---|---|
| V(content) | `msg_content ∈ {nocomm, raw, ip, arpred, dhatc, learned}` | `messages.py` content switch |
| V(ρ) | `demand.ar1_rho` | env config |
| V(geometry) | `comm_topology` → routing matrix | `messages.py` routing |
| V(β) | `beta` in the reward mix | one line in reward assembly |
| V(budget) | `budget_milestones` checkpoints | trainer snapshot list |
| do(m) probes | provider wrapper name | `evaluate.py` |

---

## 7. Diagnostics that watch the known failure modes

fig14/fig15 carry over. Two changes, both bought by this project's scar tissue:

1. **Honest explained variance**, logged at every gate: EV = 1 − Var(G−V)/Var(G) against
   Monte-Carlo returns — *not* the v5 metric whose target was defined as adv+V and whose
   residual was therefore adv by construction. A **canary**: honest EV < 0.05 after
   episode 3000 prints a loud warning in the trainlog and the report. The D1 critic
   collapse was invisible for seven tuning rounds; in v6 it would be a red line on every
   fig15 and a warning string in every report.
2. **Critic-input magnitude** (mean and max |state|/100) on the same panel, watching the
   fixed-divisor assumption directly.

---

## 8. The run-artifact contract

Every run writes one directory with a fixed schema; every consumer reads only the schema:

```
runs/<tag>/
  config_resolved.yaml     # full flattened config, hash printed at start
  command.txt              # exact argv ($-line)
  metrics_gate.csv         # per-gate: cost, best, best_ep, honest_EV, |state| stats
  metrics_update.csv       # per-update diagnostics
  ckpt_best.pt             # tensors CLONED into the payload (the 1.5e-3 drift lesson);
                           # payload carries config + env + seed (eval-parity lesson)
  ckpt_budget<N>.pt        # V(budget) snapshots
  eval/seed<K>.json        # deterministic eval dumps
  eval/eval_stdout.txt     # never DEVNULL (the c7 lesson)
```

`report.py` is fail-closed against this contract exactly as in the gate-fix: missing
reference → NO-REF, missing run → NO-RUN, missing/NaN eval → EVAL-ERROR, exit non-zero
unless all PASS, one lookup shared by decision and display, no sentinels ever.

---

## 9. Tests — the scar tissue, formalized

Written **before** the first training run; all must be green before any run is trusted.

| test | asserts | lesson it encodes |
|---|---|---|
| T-SYM | `learned`-arm agent with zeroed provider produces **bit-identical trajectories** to nocomm under CRN | arm symmetry is structural; the CRN tripwire, by design |
| T-PARAM | parameter count and shapes identical across all six contents | no arm grows anatomy |
| T-GRAD-1 | message head receives zero gradient in every content except `learned` | D1 dry-run question |
| T-GRAD-2 | no gradient reaches the frozen forecaster; incoming-message edge is detached in `learned` | A-fix, `separate_frozen` gate |
| T-ARPRED | provider output equals the closed-form AR(1) conditional mean | content correctness |
| T-INTERV | `zeroed` wrapper ≡ nocomm provider; `shuffle` preserves marginals | probe validity |
| T-EV | honest-EV computation on synthetic V,G with known answer | never again an EV that can't measure fit |
| T-FLAGS | sweep dry-run propagates every knob in §6 | the reward-scale near-miss |
| T-REPORT | fail-closed statuses and exit codes on fixtures | the c7 incident |
| **T-REPRO** | **Phase-A nocomm reproduces the reference learning curve** (band, not point) | the entire reason v6 exists |

---

## 10. Build sequence, with gates

**Phase A — core (env + agent + train, nocomm only).** Target: the reference recipe in the
v6 skeleton, S-grid action space the only deliberate deviation. **Gate:** fig14 within the
reference's band by ~8k episodes on dev seeds. If this gate fails, the defect is in ~600
lines of core and is localized — which is the entire argument for greenfield over another
tuning round. Nothing else proceeds until this passes.

**Phase B — messages.** Implement the provider + contents + topologies + wrappers; run the
test battery T-SYM…T-INTERV. **Gate:** all green, plus one `raw` dev run whose V has the
registered sign (s60 raw was +624.9 under v3 — direction, not magnitude, is the check).

**Phase C — the ladder on dev seeds.** All six contents × dev seeds under identical
adopted hyperparameters (arm symmetry non-negotiable). Produces the v6 dev validation that
the amendment requires.

**Phase D — registration.** v6 is a new instrument: amendment (R1–R4 disclosed, A19
continuity language carried), re-hash prereg, all arms retrain, pod. **No confirmatory
seed (70–94) is touched before this point** — unchanged discipline.

The amendment cost is *not* a cost of this redesign: v5 already owed a fresh dev
validation + amendment as a new instrument. v6 rides the same paperwork with a better
agent inside it.

---

## 11. Registered decisions to disclose (R-list)

| id | decision | replaces | disclosed rationale |
|---|---|---|---|
| R1 | S-grid [0,100]×41 | [0,160]×41 (QMIX-shared) | cold-start flood; QMIX unadjudicable anyway |
| R2 | MC returns | GAE(λ=0.95), vtarget=adv+V | robustness to critic pathology; unbiased |
| R3 | clip 0.1, grad-norm 0.2 | 0.2 / 0.5–10 | reference values; c4 falsified clip as a lever |
| R4 | fixed budget, no early stop | patience+floor | three incidents; budget axis needs milestones anyway |
| R5 | shared actor + role one-hot | 4 separate actors | reference design; per-role check in Phase A |
| R6 | critic ReLU/256 | Tanh/64 | reference design; probe evidence when available |

## 12. Open questions (decide before Phase A, cheap to decide now)

1. **Probe verdict pending.** If the D1 probe *refutes* Tanh saturation, R6 stays (the
   reference uses ReLU regardless) but the Phase-A failure analysis, should T-REPRO fail,
   starts elsewhere (width, target construction).
2. **Gate composition.** Keeping ρ∈{.15,.45,.75} preserves protocol continuity; the
   monitor-only 0.9 trace covers visibility. If the committee ever asks why selection
   ignores the deployment regime, the leakage-safety argument is the answer — but write it
   into the amendment text now, not under questioning.
3. **M (channel width).** Fixed by the widest content — likely `learned`'s dimension
   (v5: 3). Confirm no planned content needs more before freezing; changing M later is an
   amendment.
