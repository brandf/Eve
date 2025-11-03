#!/usr/bin/env python3
"""
Lightweight Eve hyperparameter tuner.

Runs a handful of short depth-12 training jobs (on either the h100 or rtx5090
profile) to gather signal on the best (beta1, beta2, eta) combo. Designed to be
cheap: two-stage random / local search with a few thousand iterations each.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HyperParams:
    beta1: float
    beta2: float
    eta: float

    def clamp(self) -> "HyperParams":
        return HyperParams(
            beta1=float(min(max(self.beta1, 0.80), 0.98)),
            beta2=float(min(max(self.beta2, 0.996), 0.9999)),
            eta=float(min(max(self.eta, 0.5), 1.5)),
        )

    def perturb(self, beta1_sigma=0.02, beta2_sigma=3e-4, eta_sigma=0.1) -> "HyperParams":
        return HyperParams(
            beta1=self.beta1 + random.uniform(-beta1_sigma, beta1_sigma),
            beta2=self.beta2 + random.uniform(-beta2_sigma, beta2_sigma),
            eta=self.eta + random.uniform(-eta_sigma, eta_sigma),
        ).clamp()


@dataclass
class TrialResult:
    stage: str
    trial_id: str
    params: HyperParams
    iterations: int
    min_bpb: float
    final_bpb: float
    runtime_sec: float
    command: List[str]
    log_path: Path


def ensure_datasets(min_shards: int = 16) -> None:
    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))
    data_dir = base_dir / "base_data"
    shards = list(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if len(shards) >= min_shards:
        return
    print(f"[autotune] Dataset shards missing ({len(shards)} found). Bootstrapping environment and downloading {min_shards} shards...")
    venv_activate = REPO_ROOT / ".venv" / "bin" / "activate"
    if not venv_activate.exists():
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv executable not found. Please install uv before running autotune.")
        subprocess.run([uv, "sync", "--extra", "gpu"], cwd=REPO_ROOT, check=True)
    cmd = f". {venv_activate} && python -m nanochat.dataset -n {min_shards}"
    subprocess.run(["bash", "-lc", cmd], cwd=REPO_ROOT, check=True)


def run_trial(
    params: HyperParams,
    *,
    iterations: int,
    profile: str,
    eval_tokens: int,
    run_name: str,
    stage: str,
    extra_flags: Sequence[str],
    keep_logs: bool,
) -> TrialResult:
    device_batch = 24 if profile == "h100" else 12
    total_batch = 49_152
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "scripts.base_train",
        "--",
        "--depth=12",
        f"--device_batch_size={device_batch}",
        f"--total_batch_size={total_batch}",
        f"--num_iterations={iterations}",
        f"--eval_tokens={eval_tokens}",
        "--core_metric_every=-1",
        "--sample_every=-1",
        f"--run={run_name}",
        "--model_tag=autotune_eve",
        "--eve",
        "True",
        f"--eve_beta1={params.beta1:.6f}",
        f"--eve_beta2={params.beta2:.6f}",
        f"--eve_eta={params.eta:.6f}",
        f"--eve_eps=1e-8",
    ]
    cmd.extend(extra_flags)

    env = os.environ.copy()
    env.setdefault("WANDB_RUN", "dummy")
    env.setdefault("OMP_NUM_THREADS", "1")

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runtime = time.time() - start

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = REPO_ROOT / "autotune_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    log_path.write_text(result.stdout)

    min_bpb = math.inf
    final_bpb = math.inf
    for line in result.stdout.splitlines():
        if "Validation bpb" in line:
            try:
                val = float(line.split(":")[-1].strip())
            except ValueError:
                continue
            min_bpb = min(min_bpb, val)
            final_bpb = val
    if math.isinf(min_bpb):
        print(f"[autotune] WARNING: no validation bpb found in trial {run_name}; treating as failure.")
    if result.returncode != 0:
        print(f"[autotune] Trial {run_name} failed with exit code {result.returncode}.")
        print(result.stdout)
        min_bpb = math.inf
        final_bpb = math.inf

    if not keep_logs and log_path.exists():
        log_path.unlink(missing_ok=True)

    return TrialResult(
        stage=stage,
        trial_id=run_name,
        params=params,
        iterations=iterations,
        min_bpb=min_bpb,
        final_bpb=final_bpb,
        runtime_sec=runtime,
        command=cmd,
        log_path=log_path,
    )


def sample_initial_candidates(n: int, seed: int) -> List[HyperParams]:
    random.seed(seed)
    candidates = [HyperParams(beta1=0.90, beta2=0.9990, eta=1.00)]
    while len(candidates) < n:
        hp = HyperParams(
            beta1=random.uniform(0.85, 0.95),
            beta2=random.uniform(0.9985, 0.9995),
            eta=random.uniform(0.8, 1.2),
        ).clamp()
        candidates.append(hp)
    return candidates


def select_top(results: Iterable[TrialResult], k: int) -> List[TrialResult]:
    return sorted(results, key=lambda r: r.min_bpb)[:k]


def format_result(result: TrialResult) -> str:
    params = result.params
    return (
        f"[{result.stage}] {result.trial_id}: min_bpb={result.min_bpb:.4f}, "
        f"beta1={params.beta1:.4f}, beta2={params.beta2:.6f}, eta={params.eta:.3f}, "
        f"iters={result.iterations}, runtime={result.runtime_sec/60:.1f}m"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Eve hyperparameter tuner.")
    parser.add_argument("--profile", choices=["h100", "rtx5090"], default="h100")
    parser.add_argument("--stage1-trials", type=int, default=4)
    parser.add_argument("--stage1-iters", type=int, default=2500)
    parser.add_argument("--stage2-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--eval-tokens", type=int, default=16_384)
    parser.add_argument("--extra-flag", action="append", default=[], help="Additional flag to pass through to base_train.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_datasets()

    stage1_candidates = sample_initial_candidates(args.stage1_trials, args.seed)
    results: List[TrialResult] = []

    print(f"[autotune] Stage 1: evaluating {len(stage1_candidates)} random candidates...")
    for idx, hp in enumerate(stage1_candidates, 1):
        run_name = f"autotune_stage1_{idx}"
        result = run_trial(
            hp,
            iterations=args.stage1_iters,
            profile=args.profile,
            eval_tokens=args.eval_tokens,
            run_name=run_name,
            stage="stage1",
            extra_flags=args.extra_flag,
            keep_logs=args.keep_logs,
        )
        results.append(result)
        print("  " + format_result(result))

    top_stage1 = select_top(results, k=min(2, len(results)))
    print(f"[autotune] Stage 1 best candidates:")
    for res in top_stage1:
        print("  " + format_result(res))

    stage2_candidates: List[HyperParams] = []
    for res in top_stage1:
        stage2_candidates.append(res.params)
        stage2_candidates.append(res.params.perturb(beta1_sigma=0.01, beta2_sigma=2.5e-4, eta_sigma=0.08))
        stage2_candidates.append(res.params.perturb(beta1_sigma=0.015, beta2_sigma=3e-4, eta_sigma=0.1))

    print(f"[autotune] Stage 2: refining {len(stage2_candidates)} candidates with {args.stage2_iters} iterations each...")
    for idx, hp in enumerate(stage2_candidates, 1):
        run_name = f"autotune_stage2_{idx}"
        result = run_trial(
            hp,
            iterations=args.stage2_iters,
            profile=args.profile,
            eval_tokens=args.eval_tokens,
            run_name=run_name,
            stage="stage2",
            extra_flags=args.extra_flag,
            keep_logs=args.keep_logs,
        )
        results.append(result)
        print("  " + format_result(result))

    successful = [r for r in results if math.isfinite(r.min_bpb)]
    if not successful:
        print("[autotune] All trials failed; please inspect logs in autotune_logs/.")
        sys.exit(1)

    best = min(successful, key=lambda r: r.min_bpb)
    summary = {
        "profile": args.profile,
        "best": {
            "beta1": best.params.beta1,
            "beta2": best.params.beta2,
            "eta": best.params.eta,
            "min_bpb": best.min_bpb,
            "stage": best.stage,
            "iterations": best.iterations,
        },
        "trials": [
            {
                "stage": r.stage,
                "trial_id": r.trial_id,
                "beta1": r.params.beta1,
                "beta2": r.params.beta2,
                "eta": r.params.eta,
                "min_bpb": r.min_bpb,
                "final_bpb": r.final_bpb,
                "iterations": r.iterations,
                "runtime_minutes": r.runtime_sec / 60,
                "log_path": str(r.log_path),
            }
            for r in successful
        ],
    }

    summary_path = REPO_ROOT / "autotune_logs" / "eve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[autotune] Recommended Eve settings:")
    print(f"  beta1 = {best.params.beta1:.4f}")
    print(f"  beta2 = {best.params.beta2:.6f}")
    print(f"  eta   = {best.params.eta:.3f}")
    print(f"  min bpb observed = {best.min_bpb:.4f} ({best.stage}, {best.iterations} iters)")
    print(f"  summary written to {summary_path}")


if __name__ == "__main__":
    main()
