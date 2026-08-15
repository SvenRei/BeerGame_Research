"""scripts/ii_local.py -- run a batch of training jobs on this machine.

Local execution differs from the pod in one way that matters: without thread pinning,
every job grabs every core and they thrash. On the pod each job used ~4.7 cores
happily because there were 85 of them; on a laptop that is the difference between
finishing overnight and not finishing.

    from ii_local import run_jobs
    run_jobs(jobs, workers=6)

Each job is a dict with 'tag' and 'set' (the --set string). Jobs whose run directory
already contains ckpt_best.pt and whose log records completion are skipped, so an
interrupted batch can simply be re-run.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _done(tag):
    ck = os.path.join(ROOT, "runs", tag, "ckpt_best.pt")
    lg = os.path.join(ROOT, "runs", "logs", f"{tag}.log")
    if not (os.path.exists(ck) and os.path.exists(lg)):
        return False
    try:
        return "[signal] done." in open(lg, encoding="utf-8", errors="ignore").read()
    except OSError:
        return False


def check_unique(jobs):
    """The SIGNAL-I failure that cost a topology cell: two conditions generated the same
    tag, both were queued because the idempotency guard runs before either has written
    anything, and the pool ran them concurrently into one directory. Assert uniqueness
    at generation time -- the only place it can be caught."""
    seen, dup = set(), set()
    for j in jobs:
        if j["tag"] in seen:
            dup.add(j["tag"])
        seen.add(j["tag"])
    if dup:
        raise SystemExit(f"FAIL-CLOSED: duplicate tags in job list: {sorted(dup)}. "
                         f"Two conditions would train into one directory.")
    return True


def run_jobs(jobs, workers=6, threads=1, dry=False):
    check_unique(jobs)
    os.makedirs(os.path.join(ROOT, "runs", "logs"), exist_ok=True)
    todo = [j for j in jobs if not _done(j["tag"])]
    print(f"[local] {len(jobs)} jobs, {len(jobs) - len(todo)} already complete, "
          f"{len(todo)} to run, {workers} at a time, {threads} thread(s) each")
    if dry:
        for j in todo[:5]:
            print("   ", j["tag"], "|", j["set"][:110])
        print(f"    ... ({len(todo)} total)")
        return

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[k] = str(threads)

    t0 = time.time()
    for i in range(0, len(todo), workers):
        batch = todo[i:i + workers]
        procs = []
        for j in batch:
            log = os.path.join(ROOT, "runs", "logs", f"{j['tag']}.log")
            cmd = [sys.executable, "-m", "signal_lab.train", "--set"] + j["set"].split()
            fh = open(log, "w", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                           env=env), fh, j["tag"]))
        for p, fh, tag in procs:
            p.wait(); fh.close()
            print(f"   {'ok ' if p.returncode == 0 else 'FAIL'} {tag}")
        el = time.time() - t0
        done = min(i + workers, len(todo))
        rate = el / max(1, done)
        print(f"   -- {done}/{len(todo)} in {el / 60:.0f} min, "
              f"~{rate * (len(todo) - done) / 60:.0f} min remaining")
    print(f"[local] finished in {(time.time() - t0) / 60:.0f} min")


def verify(tags):
    """Post-run integrity: completion, tracebacks, and the critic canary."""
    import csv
    bad = []
    for t in tags:
        lg = os.path.join(ROOT, "runs", "logs", f"{t}.log")
        gt = os.path.join(ROOT, "runs", t, "metrics_gate.csv")
        txt = open(lg, encoding="utf-8", errors="ignore").read() if os.path.exists(lg) else ""
        done = "[signal] done." in txt
        tb = "Traceback" in txt
        ev = None
        if os.path.exists(gt):
            vals = [float(r["honest_ev"]) for r in csv.DictReader(open(gt, encoding="utf-8"))
                    if r.get("honest_ev") not in (None, "", "nan")
                    and float(r.get("episode", 0)) >= 3000]
            ev = min(vals) if vals else None
        ok = done and not tb and (ev is None or ev >= 0.05)
        if not ok:
            bad.append((t, done, tb, ev))
    print(f"[verify] {len(tags) - len(bad)}/{len(tags)} clean")
    for t, done, tb, ev in bad:
        print(f"   {t}: done={done} traceback={tb} min_honest_EV={ev}")
    return not bad
