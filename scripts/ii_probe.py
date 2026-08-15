"""scripts/ii_probe.py -- what knobs does this codebase actually have?

SIGNAL-II needs three things SIGNAL-I never varied: the action-grid bounds, the lead
time, and the echelon count. Before a single job is queued, this script finds out what
those parameters are CALLED and whether they are settable, rather than assuming names
that may not exist. SIGNAL-I's action-ceiling defect came from assuming a grid was wide
enough without checking; the same class of mistake at the planning stage is cheaper to
avoid than to discover after twenty hours of compute.

    python scripts/ii_probe.py

Writes docs/II_PROBE.json with what it found, so the later scripts can build commands
from discovered names instead of guesses.
"""
import argparse
import inspect
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Names a parameter might plausibly carry. The probe reports which ones EXIST; it never
# picks one silently.
CANDIDATES = {
    "action_grid_max": ["s_grid_max", "action_max", "s_max", "grid_max", "max_s",
                        "s_grid_hi", "order_up_to_max", "a_max"],
    "action_grid_bins": ["s_grid_bins", "n_actions", "action_bins", "grid_bins",
                         "n_bins", "s_levels", "action_levels"],
    "lead_time": ["lead_time", "leadtime", "L", "ship_delay", "transport_delay",
                  "lead", "delay"],
    "n_echelons": ["n_agents", "n_stages", "n_echelons", "N", "stages", "echelons"],
    "warmup": ["warmup_episodes", "warmup", "cold_start_episodes", "cold_start"],
    "init_bias": ["init_action_bias", "action_init", "head_init", "init_logit_bias"],
}


def _load_yaml(path):
    """Read conf/signal.yaml without requiring pyyaml: the file is flat key: value."""
    out = {}
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*(#.*)?$", ln)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _module_constants(modname):
    try:
        mod = __import__(modname, fromlist=["*"])
    except Exception as e:                       # noqa: BLE001 - report, never crash
        return {"__error__": f"{type(e).__name__}: {e}"}
    out = {}
    for k, v in vars(mod).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, str, bool, list, tuple)):
            out[k] = v if not isinstance(v, (list, tuple)) else list(v)[:8]
    return out


def _signature(modname, objname):
    try:
        mod = __import__(modname, fromlist=[objname])
        obj = getattr(mod, objname)
        return str(inspect.signature(obj))
    except Exception as e:                       # noqa: BLE001
        return f"({type(e).__name__}: {e})"


def _grep(path, pattern, n=6):
    if not os.path.exists(path):
        return [f"(missing: {path})"]
    hits = []
    for i, ln in enumerate(open(path, encoding="utf-8", errors="ignore"), 1):
        if re.search(pattern, ln):
            hits.append(f"{i}: {ln.rstrip()[:150]}")
            if len(hits) >= n:
                break
    return hits or ["(no match)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "II_PROBE.json"))
    a = ap.parse_args()

    rep = {"root": ROOT}

    print("=" * 78)
    print("1. conf/signal.yaml -- declared configuration")
    print("=" * 78)
    cfg = _load_yaml(os.path.join(ROOT, "conf", "signal.yaml"))
    rep["config_keys"] = sorted(cfg)
    if not cfg:
        print("  (conf/signal.yaml not found or unreadable)")
    for group, names in CANDIDATES.items():
        found = [n for n in names if n in cfg]
        print(f"  {group:18s} -> {found if found else 'NOT FOUND under any candidate name'}")
        if found:
            for n in found:
                print(f"    {n} = {cfg[n]}")
    rep["config_matches"] = {g: [n for n in names if n in cfg]
                             for g, names in CANDIDATES.items()}

    print()
    print("=" * 78)
    print("2. env/beer_game.py -- module constants and the grid/lead-time definition")
    print("=" * 78)
    rep["env_constants"] = _module_constants("env.beer_game")
    for k, v in list(rep["env_constants"].items())[:24]:
        print(f"  {k:24s} = {v}")
    print()
    print("  lines mentioning lead time / delay / pipeline:")
    for ln in _grep(os.path.join(ROOT, "env", "beer_game.py"),
                    r"lead|delay|pipeline|ship", 8):
        print("   ", ln)
    print()
    print("  lines mentioning max_order or action bounds:")
    for ln in _grep(os.path.join(ROOT, "env", "beer_game.py"),
                    r"max_order|action_space|clip", 6):
        print("   ", ln)

    print()
    print("=" * 78)
    print("3. signal_lab/agent.py -- where the S-grid is defined")
    print("=" * 78)
    for ln in _grep(os.path.join(ROOT, "signal_lab", "agent.py"),
                    r"linspace|grid|n_actions|bins|Categorical|logits", 10):
        print("   ", ln)
    rep["agent_constants"] = _module_constants("signal_lab.agent")

    print()
    print("=" * 78)
    print("4. signal_lab/train.py -- what --set accepts")
    print("=" * 78)
    for ln in _grep(os.path.join(ROOT, "signal_lab", "train.py"),
                    r"--set|ALLOWED|WHITELIST|unknown key|cfg\[|def main", 10):
        print("   ", ln)

    print()
    print("=" * 78)
    print("5. baselines.py -- does it take a lead time?")
    print("=" * 78)
    for ln in _grep(os.path.join(ROOT, "signal_lab", "baselines.py"),
                    r"add_argument|lead|cfg = \{", 12):
        print("   ", ln)

    print()
    print("=" * 78)
    print("6. BeerGame constructor")
    print("=" * 78)
    print("  BeerGame" + _signature("env.beer_game", "BeerGame"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print()
    print(f"[probe] wrote {a.out}")
    print()
    print("WHAT TO DO WITH THIS: the three parameters SIGNAL-II must vary are the action")
    print("grid bounds, the lead time and the echelon count. For each, either a settable")
    print("config key exists -- in which case the later scripts use that name -- or it is")
    print("hardcoded, in which case it must be lifted into the config BEFORE calibration.")
    print("Paste this output back before running ii_fit_benchmarks.py.")


if __name__ == "__main__":
    main()
