#!/usr/bin/env python3
import argparse
import math
import numpy as np
import torch

from phase3_ideal_observer import (
    CLUTTER_SEEDS,
    TARGET_SIZE_M,
    NSTEP,
    DEFAULT_QUAD,
    base_manifest_from_seed,
    frozen_locations,
    make_specs,
    simulate_schedules,
    schedule_bank,
)

MASTER_SEED = 9_050_026

def cargo_class_evidence(bank):
    L = int(bank["tungsten"].numel())
    logL = math.log(L)
    e1 = torch.logsumexp(bank["tungsten"], dim=0) - logL
    steel = torch.logsumexp(bank["steel"], dim=0) - logL
    e0 = torch.logsumexp(torch.stack([
        bank["no_target"] + math.log(0.5),
        steel + math.log(0.5),
    ]), dim=0)
    return e1, e0

def posterior_true(e, true_idx):
    z = torch.logsumexp(e, dim=0)
    return float(torch.exp(e[true_idx] - z))

def gap_true_vs_best_wrong(e, true_idx):
    mask = torch.ones_like(e, dtype=torch.bool)
    mask[true_idx] = False
    return float(e[true_idx] - e[mask].max())

def run_scene(spec, bases, centers, budget, dev, nq, track_chunk, loc_chunk):
    schedules = simulate_schedules(spec, bases, centers, budget, NSTEP, dev, MASTER_SEED)
    out = {}
    for sched_name, (sched_events, _) in schedules.items():
        e1s, e0s = [], []
        for base in bases:
            bank = schedule_bank(
                sched_events, base, centers, TARGET_SIZE_M,
                NSTEP, nq, dev, track_chunk, loc_chunk
            )
            e1, e0 = cargo_class_evidence(bank)
            e1s.append(e1)
            e0s.append(e0)
        e1 = torch.stack(e1s)
        e0 = torch.stack(e0s)
        ci = spec.cargo_idx
        known = float(e1[ci] - e0[ci])
        unknown = float(torch.logsumexp(e1, 0) - torch.logsumexp(e0, 0))
        out[sched_name] = {
            "known": known,
            "unknown": unknown,
            "delta": unknown - known,
            "h1_gap": gap_true_vs_best_wrong(e1, ci),
            "h0_gap": gap_true_vs_best_wrong(e0, ci),
            "h1_true_post": posterior_true(e1, ci),
            "h0_true_post": posterior_true(e0, ci),
            "best_h1_cargo": int(torch.argmax(e1)),
            "best_h0_cargo": int(torch.argmax(e0)),
        }
    return out

def summarize(rows, sched):
    vals = [r[sched] for r in rows]
    print(f"\n=== {sched} ===")
    for key in ["delta", "h1_gap", "h0_gap", "h1_true_post", "h0_true_post"]:
        a = np.array([v[key] for v in vals], dtype=float)
        print(f"{key:14s} min={a.min(): .6g} median={np.median(a): .6g} max={a.max(): .6g}")
    n_h1 = sum(v["best_h1_cargo"] == rows[i]["true_cargo"] for i, v in enumerate(vals))
    n_h0 = sum(v["best_h0_cargo"] == rows[i]["true_cargo"] for i, v in enumerate(vals))
    print(f"true cargo is H1 argmax: {n_h1}/{len(vals)}")
    print(f"true cargo is H0 argmax: {n_h0}/{len(vals)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-benign", type=int, default=10)
    ap.add_argument("--n-threat", type=int, default=10)
    ap.add_argument("--budget", type=int, default=50_000)
    ap.add_argument("--quadrature", type=int, default=DEFAULT_QUAD)
    ap.add_argument("--track-chunk", type=int, default=2500)
    ap.add_argument("--loc-chunk", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    bases = [base_manifest_from_seed(s) for s in CLUTTER_SEEDS]
    centers = frozen_locations()

    benign = make_specs(
        args.n_benign, 0, MASTER_SEED + 200,
        len(bases), len(centers), 2_000_000
    )
    threat = make_specs(
        args.n_threat, 1, MASTER_SEED + 300,
        len(bases), len(centers), 3_000_000
    )
    specs = benign + threat
    rows = []

    print(f"device: {dev}")
    print(f"budget: {args.budget:,}")
    print(f"cargo hypotheses: {len(bases)}")
    print(f"locations: {len(centers)}")
    print(f"scenes: {len(benign)} benign + {len(threat)} threat")

    for i, spec in enumerate(specs, 1):
        res = run_scene(
            spec, bases, centers, args.budget, dev,
            args.quadrature, args.track_chunk, args.loc_chunk
        )
        row = {"label": spec.label, "true_cargo": spec.cargo_idx, **res}
        rows.append(row)
        s = res["static"]
        print(
            f"{i:3d}/{len(specs)} label={spec.label} cargo={spec.cargo_idx:2d} "
            f"static d={s['delta']:+.3e} "
            f"H1gap={s['h1_gap']:.2f} H0gap={s['h0_gap']:.2f} "
            f"Ptrue(H1)={s['h1_true_post']:.6f} "
            f"Ptrue(H0)={s['h0_true_post']:.6f}"
        )

    summarize(rows, "static")
    summarize(rows, "two_view")

    print("\nINTERPRETATION")
    print(
        "If d is zero/near-zero while true-cargo posterior masses are ~1 "
        "and true-vs-wrong gaps are very large, Phase 3B collapses numerically "
        "to Phase 3A because the raw data identify the cargo state inside the "
        "closed 16-state library almost perfectly."
    )

if __name__ == "__main__":
    main()
