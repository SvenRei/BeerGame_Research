"""sweep.py -- arm matrix -> jobs. The whole hypothesis ladder is this file's CLI.

  python sweep.py --contents nocomm,raw --seeds 60,61 --rho 0.9 --episodes 8000
  python sweep.py --contents nocomm --seeds 60 --set beta=0.5 --suffix b05
  python sweep.py ... --dry-run          # print the exact child commands, train nothing

Every job is one signal_lab/train.py invocation with explicit --set overrides; the
full command line is teed as a '$' line into runs/logs/.trainlog_<tag>.txt.
Completion invariant: the trainer prints "[signal] done." on every clean exit -- its
absence means truncation/crash, full stop (no milestone-string inference, the lesson
from the legacy runner).
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_jobs(a):
    jobs = []
    for content, seed in itertools.product(a.contents.split(","), a.seeds.split(",")):
        content, seed = content.strip(), int(seed)
        tag = f"{content}{('_' + a.suffix) if a.suffix else ''}_s{seed}"
        overrides = [f"content={content}", f"seed={seed}", f"tag={tag}",
                     f"rho={a.rho}", f"total_episodes={a.episodes}"]
        overrides += list(a.set or [])
        cmd = [sys.executable, os.path.join(ROOT, "signal_lab", "train.py"),
               "--config", a.config, "--set", *overrides]
        jobs.append((tag, cmd))
    return jobs


def run_job(tag, cmd):
    log_dir = os.path.join(ROOT, "runs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f".trainlog_{tag}.txt")
    with open(log, "w", buffering=1) as f:
        f.write("$ " + " ".join(cmd) + "\n")
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    done = "[signal] done." in open(log, errors="ignore").read()
    return tag, ("ok" if (r.returncode == 0 and done) else
                 f"FAIL rc={r.returncode} complete={done} (see {log})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--contents", default="nocomm")
    ap.add_argument("--seeds", default="60,61")
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--config", default=os.path.join(ROOT, "conf", "signal.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra config overrides forwarded to every job")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", dest="dry",
                    help="print each child command and exit WITHOUT training -- verify "
                         "every flag propagated before spending hours")
    a = ap.parse_args(argv)
    jobs = build_jobs(a)
    print(f"== sweep: {len(jobs)} job(s), rho={a.rho}, episodes={a.episodes}, "
          f"workers={a.workers} ==")
    if a.dry:
        for tag, cmd in jobs:
            print(f"  DRY-RUN (not executed): {' '.join(cmd)}")
        print("dry run: nothing trained. Drop --dry-run to execute.")
        return 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, status in ex.map(lambda j: run_job(*j), jobs):
            print(f"  {tag}: {status}")
    print("next: python -m signal_lab.evaluate --ckpt runs/<tag>/ckpt_best.pt ; "
          "python -m signal_lab.report --arms <tags>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
