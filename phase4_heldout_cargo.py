#!/usr/bin/env python3
"""
phase4_heldout_cargo.py
Frozen Phase 4A held-out-cargo generalization gate.
"""

import argparse, hashlib, json, subprocess, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from phase3_ideal_observer import (
    CLUTTER_SEEDS, TARGET_SIZE_M, THREAT_MAT, CONFUSER_MAT,
    NSTEP, DEFAULT_QUAD,
    base_manifest_from_seed, frozen_locations, insert_cube,
    rotate_manifest_y90, simulate, schedule_bank, scene_scores_from_banks,
    auc_tie_correct, bootstrap_auc_ci, bootstrap_auc_delta_ci,
    operating_point,
)

PRIMARY_BUDGET = 500
MASTER_SEED = 9_060_026
CAL_START = 40_000
NEG_START = 50_000
POS_START = 60_000

@dataclass(frozen=True)
class Spec:
    label: int
    cargo_seed: int
    probe_loc_idx: int
    benign_kind: str
    realization_id: int

def make_specs(n, label, cargo_start, stream_seed, nloc, rid_offset):
    rng = np.random.default_rng(stream_seed)
    out = []
    for i in range(n):
        loc = int(rng.integers(0, nloc))
        kind = "threat" if label else ("steel" if rng.random() < 0.5 else "none")
        out.append(Spec(label, cargo_start+i, loc, kind, rid_offset+i))
    return out

def true_manifest(spec, centers):
    base = base_manifest_from_seed(spec.cargo_seed)
    c = centers[spec.probe_loc_idx]
    if spec.label:
        return insert_cube(base, THREAT_MAT, c)
    if spec.benign_kind == "steel":
        return insert_cube(base, CONFUSER_MAT, c)
    return insert_cube(base, None, c)

def seed_pair(realization_id, schedule_id, view_id):
    x = MASTER_SEED + realization_id*1_000_003 + schedule_id*10_007 + view_id*101
    return int(x+17), int(x+53)

def simulate_schedules(spec, centers, budget, nstep, dev):
    m = true_manifest(spec, centers)

    ss, sc = seed_pair(spec.realization_id, 0, 0)
    ev_static, inc_static = simulate(m, budget, nstep, dev, ss, sc)

    nA = budget // 2
    nB = budget - nA
    ss, sc = seed_pair(spec.realization_id, 1, 0)
    ev_A, inc_A = simulate(m, nA, nstep, dev, ss, sc)

    mr = rotate_manifest_y90(m)
    ss, sc = seed_pair(spec.realization_id, 1, 1)
    ev_B, inc_B = simulate(mr, nB, nstep, dev, ss, sc)

    return {
        "static": ([("A", ev_static)], inc_static),
        "two_view": ([("A", ev_A), ("B", ev_B)], inc_A + inc_B),
    }

def eval_scene(spec, observer_bases, centers, schedules, nstep, nq, dev, track_chunk, loc_chunk):
    out = {}
    for name, (events, incident) in schedules.items():
        banks = [
            schedule_bank(events, base, centers, TARGET_SIZE_M,
                          nstep, nq, dev, track_chunk, loc_chunk)
            for base in observer_bases
        ]
        blind, revealed = scene_scores_from_banks(banks, spec.probe_loc_idx)
        out[name] = {"blind": blind, "revealed": revealed, "incident": int(incident)}
    return out

def sha256_file(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def git_head():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        return None

def score_summary(pos, neg):
    p = np.asarray(pos, float); n = np.asarray(neg, float)
    return {
        "neg_q01": float(np.quantile(n,.01)),
        "neg_q50": float(np.quantile(n,.50)),
        "neg_q99": float(np.quantile(n,.99)),
        "pos_q01": float(np.quantile(p,.01)),
        "pos_q50": float(np.quantile(p,.50)),
        "pos_q99": float(np.quantile(p,.99)),
        "hard_margin": float(p.min()-n.max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    ap.add_argument("--n-cal-benign", type=int, default=200)
    ap.add_argument("--n-test-benign", type=int, default=200)
    ap.add_argument("--n-test-threat", type=int, default=200)
    ap.add_argument("--nstep", type=int, default=NSTEP)
    ap.add_argument("--quadrature", type=int, default=DEFAULT_QUAD)
    ap.add_argument("--track-chunk", type=int, default=2500)
    ap.add_argument("--loc-chunk", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="phase4a_heldout_cargo.npz")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.budget = 100
        args.n_cal_benign = args.n_test_benign = args.n_test_threat = 3
        observer_seeds = CLUTTER_SEEDS[:2]
        centers = frozen_locations()[[13,4,10,16,22]]
        print("WARNING: --smoke mode. Results are NON-SCIENTIFIC.")
    else:
        observer_seeds = CLUTTER_SEEDS
        centers = frozen_locations()

    dev = torch.device(args.device)
    observer_bases = [base_manifest_from_seed(s) for s in observer_seeds]

    print(f"device: {dev} torch {torch.__version__}")
    print("phase: 4A held-out cargo")
    print(f"budget: {args.budget:,}")
    print(f"observer cargo hypotheses: {len(observer_bases)}")
    print(f"locations: {len(centers)}")

    cal_specs = make_specs(args.n_cal_benign, 0, CAL_START, MASTER_SEED+100, len(centers), 4_000_000)
    neg_specs = make_specs(args.n_test_benign, 0, NEG_START, MASTER_SEED+200, len(centers), 5_000_000)
    pos_specs = make_specs(args.n_test_threat, 1, POS_START, MASTER_SEED+300, len(centers), 6_000_000)

    true_seeds = [s.cargo_seed for s in cal_specs+neg_specs+pos_specs]
    assert len(true_seeds) == len(set(true_seeds))
    assert set(observer_seeds).isdisjoint(true_seeds)

    print(f"observer/true cargo overlap: 0")
    print(f"cal cargo: {cal_specs[0].cargo_seed}-{cal_specs[-1].cargo_seed}")
    print(f"neg cargo: {neg_specs[0].cargo_seed}-{neg_specs[-1].cargo_seed}")
    print(f"pos cargo: {pos_specs[0].cargo_seed}-{pos_specs[-1].cargo_seed}")

    splits = [("cal_benign",cal_specs),("test_benign",neg_specs),("test_threat",pos_specs)]
    scores = {name:{k:[] for k in [
        "static_blind","static_revealed","static_incident",
        "two_blind","two_revealed","two_incident"]} for name,_ in splits}

    total = sum(len(s) for _,s in splits)
    done=0; t0=time.time()

    for split, specs in splits:
        for spec in specs:
            sched = simulate_schedules(spec, centers, args.budget, args.nstep, dev)
            res = eval_scene(spec, observer_bases, centers, sched,
                             args.nstep, args.quadrature, dev,
                             args.track_chunk, args.loc_chunk)
            scores[split]["static_blind"].append(res["static"]["blind"])
            scores[split]["static_revealed"].append(res["static"]["revealed"])
            scores[split]["static_incident"].append(res["static"]["incident"])
            scores[split]["two_blind"].append(res["two_view"]["blind"])
            scores[split]["two_revealed"].append(res["two_view"]["revealed"])
            scores[split]["two_incident"].append(res["two_view"]["incident"])
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} {time.time()-t0:8.1f}s")

    cal=scores["cal_benign"]; neg=scores["test_benign"]; pos=scores["test_threat"]

    results={}
    print("\nBLIND HELD-OUT-CARGO OBSERVER")
    for name,key in [("static","static_blind"),("two_view","two_blind")]:
        auc=auc_tie_correct(pos[key],neg[key])
        ci=bootstrap_auc_ci(pos[key],neg[key])
        th,sens,rfpr=operating_point(cal[key],pos[key],neg[key])
        summ=score_summary(pos[key],neg[key])
        results[name]={"auc":auc,"ci":ci,"threshold":th,"sensitivity":sens,"fpr":rfpr,"score_summary":summ}
        print(f"{name:>10} AUC={auc:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] sens@5%FPR={sens:.3f} testFPR={rfpr:.3f}")

    delta=results["two_view"]["auc"]-results["static"]["auc"]
    dci=bootstrap_auc_delta_ci(pos["static_blind"],neg["static_blind"],pos["two_blind"],neg["two_blind"])
    print(f"\ntwo_view-static AUC={delta:+.3f} CI=[{dci[0]:+.3f},{dci[1]:+.3f}]")

    print("\nSCORE DISTRIBUTIONS")
    for name in ["static","two_view"]:
        s=results[name]["score_summary"]
        print(f"{name:>10} benign q01/q50/q99={s['neg_q01']:.3f}/{s['neg_q50']:.3f}/{s['neg_q99']:.3f} "
              f"threat q01/q50/q99={s['pos_q01']:.3f}/{s['pos_q50']:.3f}/{s['pos_q99']:.3f} "
              f"hard_margin={s['hard_margin']:+.3f}")

    print("\nLOCATION-REVEALED")
    revealed={}
    for name,key in [("static","static_revealed"),("two_view","two_revealed")]:
        auc=auc_tie_correct(pos[key],neg[key]); ci=bootstrap_auc_ci(pos[key],neg[key],seed=54321)
        revealed[name]={"auc":auc,"ci":ci}
        print(f"{name:>10} AUC={auc:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}]")

    if not args.smoke:
        s_auc=results["static"]["auc"]; t_auc=results["two_view"]["auc"]
        if s_auc >= .90:
            decision="PASS"
        elif s_auc < .90 and delta >= .05 and dci[0] > 0 and results["two_view"]["sensitivity"] > results["static"]["sensitivity"]:
            decision="GEOMETRY SIGNAL"
        elif s_auc < .80 and t_auc < .80:
            decision="STOP"
        else:
            decision="INCONCLUSIVE"
        print(f"\nPHASE 4A DECISION: {decision}")

    metadata={
        "phase":"4A","budget":args.budget,"observer_cargo_seeds":list(observer_seeds),
        "cal_range":[cal_specs[0].cargo_seed,cal_specs[-1].cargo_seed],
        "neg_range":[neg_specs[0].cargo_seed,neg_specs[-1].cargo_seed],
        "pos_range":[pos_specs[0].cargo_seed,pos_specs[-1].cargo_seed],
        "protocol_sha256":sha256_file("PHASE4_PROTOCOL.md"),
        "git_head":git_head(),"results":results,"revealed":revealed,
        "auc_delta":delta,"auc_delta_ci":dci,
    }

    if args.save:
        arrays={"metadata_json":np.array(json.dumps(metadata))}
        for split,dd in scores.items():
            for k,v in dd.items():
                arrays[f"{split}_{k}"]=np.asarray(v)
        np.savez_compressed(args.save,**arrays)
        print(f"saved -> {args.save}")

    print("protocol sha256:",metadata["protocol_sha256"])
    print("git HEAD:",metadata["git_head"])

if __name__=="__main__":
    main()
