#!/usr/bin/env python3
"""
phase3_ideal_observer.py

Raw-event ideal-observer gate for muon-tomography-lab.

This is NOT another PoCA score. It asks how much threat-vs-benign information
exists in the simulator's raw incoming/outgoing track measurements when
momentum and target location are hidden.

Primary scientific protocol: PHASE3_PROTOCOL.md

Stage 3A (run first):
    observer knows the benign cargo packing but not target location.

Stage 3B (only if 3A justifies it):
    observer marginalizes over the frozen cargo library as well as location.

The observer uses the same midpoint path discretization and Highland covariance
structure as scene_physics.forward(), but integrates over the simulator's
latent momentum distribution rather than reading ev["p"].

Examples:

    # Non-scientific implementation check
    python phase3_ideal_observer.py --smoke

    # Frozen Phase-3A primary gate
    python phase3_ideal_observer.py \
        --mode known \
        --budget 50000 \
        --n-cal-benign 200 \
        --n-test-benign 200 \
        --n-test-threat 200 \
        --save phase3a_ideal_observer.npz

    # Phase 3B only if Phase 3A warrants it
    python phase3_ideal_observer.py \
        --mode unknown \
        --budget 50000 \
        --n-cal-benign 200 \
        --n-test-benign 200 \
        --n-test-threat 200 \
        --save phase3b_ideal_observer.npz

IMPORTANT:
  * --smoke results are never scientific evidence.
  * Full-run deviations from PHASE3_PROTOCOL.md must be recorded as protocol
    deviations rather than silently interpreted as the frozen gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from scene_physics import X0_M, inv_X0_at, make_manifest, simulate


# ---------------------------------------------------------------------------
# Frozen challenge definition
# ---------------------------------------------------------------------------

CLUTTER_SEEDS = tuple(range(30_000, 30_016))
TARGET_SIZE_M = 0.10
THREAT_MAT = "tungsten"
CONFUSER_MAT = "steel"
GRID_LEVELS_M = (-0.30, 0.0, 0.30)
PRIMARY_BUDGET = 50_000
NSTEP = 100
DEFAULT_QUAD = 16
LOG2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class SceneSpec:
    label: int                # 1 threat, 0 benign
    cargo_idx: int
    probe_loc_idx: int
    benign_kind: str          # "threat", "steel", or "none"
    realization_id: int


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _clone_box(b):
    return dict(c=b["c"].clone(), h=b["h"].clone(),
                mat=b["mat"], kind=b.get("kind", "clutter"))


def base_manifest_from_seed(seed: int, side: float = 1.0):
    """Keep only the clutter stream from make_manifest()."""
    m = make_manifest(seed, side)
    boxes = [_clone_box(b) for b in m["boxes"]]
    fill = sum(float((2 * b["h"]).prod()) for b in boxes) / side**3
    return dict(
        scene_seed=seed,
        side=side,
        boxes=boxes,
        target=None,
        present=False,
        target_mat=None,
        target_size=TARGET_SIZE_M,
        fill=fill,
        n_clutter=len(boxes),
    )


def insert_cube(base, material: str | None, center, size=TARGET_SIZE_M):
    m = dict(base)
    m["boxes"] = [_clone_box(b) for b in base["boxes"]]
    m["target_size"] = float(size)
    if material is None:
        m["target"] = None
        m["target_mat"] = None
        m["present"] = False
        return m

    c = torch.as_tensor(center, dtype=torch.float64).cpu()
    h = torch.full((3,), size / 2.0, dtype=torch.float64)
    m["target"] = dict(c=c, h=h, mat=material, kind="target")
    m["target_mat"] = material
    m["present"] = material == THREAT_MAT
    return m


def rotate_vec_y90(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return torch.stack([v[..., 2], v[..., 1], -v[..., 0]], dim=-1)


def rotate_halfsize_y90(h):
    h = torch.as_tensor(h, dtype=torch.float64)
    return torch.stack([h[..., 2], h[..., 1], h[..., 0]], dim=-1)


def rotate_manifest_y90(m):
    """Ideal complementary-view diagnostic: rotate cargo +90 deg about y."""
    out = dict(m)
    out["boxes"] = []
    for b in m["boxes"]:
        out["boxes"].append(dict(
            c=rotate_vec_y90(b["c"]).cpu(),
            h=rotate_halfsize_y90(b["h"]).cpu(),
            mat=b["mat"],
            kind=b.get("kind", "clutter"),
        ))
    if m["target"] is None:
        out["target"] = None
    else:
        b = m["target"]
        out["target"] = dict(
            c=rotate_vec_y90(b["c"]).cpu(),
            h=rotate_halfsize_y90(b["h"]).cpu(),
            mat=b["mat"],
            kind="target",
        )
    return out


def frozen_locations():
    pts = []
    for x in GRID_LEVELS_M:
        for y in GRID_LEVELS_M:
            for z in GRID_LEVELS_M:
                pts.append((x, y, z))
    return torch.tensor(pts, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Event observables
# ---------------------------------------------------------------------------

def event_observables(ev, side: float):
    """Recover transverse angular kicks/displacements without true momentum."""
    e = ev["entry"].double()
    d = ev["d_in"].double()
    x = ev["exit"].double()
    do = ev["d_out"].double()

    t_exit = (e[:, 2] + side / 2.0) / (-d[:, 2])
    ideal_exit = e + t_exit[:, None] * d

    a = torch.zeros_like(d)
    a[:, 0] = 1.0
    swap = d[:, 0].abs() > 0.9
    a[swap] = torch.tensor([0.0, 1.0, 0.0],
                           device=d.device, dtype=torch.float64)
    xh = torch.cross(a, d, dim=1)
    xh = xh / xh.norm(dim=1, keepdim=True)
    yh = torch.cross(d, xh, dim=1)

    disp = x - ideal_exit
    dx = (disp * xh).sum(1)
    dy = (disp * yh).sum(1)

    # forward() normalizes d + tx*xh + ty*yh. Dividing by the longitudinal
    # component recovers the pre-normalization tx,ty up to roundoff.
    qd = (do * d).sum(1).clamp_min(1e-15)
    tx = (do * xh).sum(1) / qd
    ty = (do * yh).sum(1) / qd

    return dict(entry=e, d=d, t_exit=t_exit,
                tx=tx, ty=ty, dx=dx, dy=dy)


# ---------------------------------------------------------------------------
# Momentum quadrature and Gaussian likelihood
# ---------------------------------------------------------------------------

def momentum_quadrature(nq: int, dev):
    """Integrate y~U(0,1), p=1000*y^(-1/1.7), using Gauss-Legendre."""
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    y = (nodes + 1.0) / 2.0
    w = weights / 2.0
    p = 1000.0 * y ** (-1.0 / 1.7)
    return (
        torch.tensor(p, device=dev, dtype=torch.float64),
        torch.log(torch.tensor(w, device=dev, dtype=torch.float64)),
    )


def marginal_event_logpdf(J0, J1, J2, obs, pgrid, logw):
    """
    J arrays may be [tracks] or [hypotheses, tracks]. Returns the same
    leading shape, with latent momentum integrated out.
    """
    J0 = J0.clamp_min(1e-14)
    logf = (1.0 + 0.038 * torch.log(J0)).clamp_min(0.0)
    lf2 = logf * logf

    k00 = (lf2 * J0).clamp_min(1e-18)
    k01 = lf2 * J1
    k11 = (lf2 * J2).clamp_min(1e-18)
    det = (k00 * k11 - k01 * k01).clamp_min(1e-30)

    tx, ty, dx, dy = obs["tx"], obs["ty"], obs["dx"], obs["dy"]
    while tx.ndim < J0.ndim:
        tx = tx.unsqueeze(0)
        ty = ty.unsqueeze(0)
        dx = dx.unsqueeze(0)
        dy = dy.unsqueeze(0)

    qx = (k11 * tx * tx - 2.0 * k01 * tx * dx + k00 * dx * dx) / det
    qy = (k11 * ty * ty - 2.0 * k01 * ty * dy + k00 * dy * dy) / det
    qk = qx + qy
    logdetK = torch.log(det)

    s = (13.6 / pgrid) ** 2
    lp = (
        -2.0 * LOG2PI
        - 2.0 * torch.log(s)
        - logdetK[..., None]
        - 0.5 * qk[..., None] / s
        + logw
    )
    return torch.logsumexp(lp, dim=-1)


# ---------------------------------------------------------------------------
# Efficient hypothesis-bank likelihood
# ---------------------------------------------------------------------------

def likelihood_bank(ev, base_manifest, centers, target_size, nstep, nq, dev,
                    track_chunk=2500, loc_chunk=6):
    """
    Return log likelihoods under one base cargo packing:
      no_target: scalar
      tungsten : [n_locations]
      steel    : [n_locations]

    The base material field is evaluated once per event chunk. Candidate cubes
    then override the base material only at their declared locations, matching
    the target-last material rule without rerunning the full material lookup
    separately for every target location.
    """
    side = float(base_manifest["side"])
    obs_all = event_observables(ev, side)
    pgrid, logw = momentum_quadrature(nq, dev)

    centers = centers.to(dev).double()
    nloc = centers.shape[0]
    half = float(target_size) / 2.0

    ll_none = torch.zeros((), device=dev, dtype=torch.float64)
    ll_w = torch.zeros(nloc, device=dev, dtype=torch.float64)
    ll_s = torch.zeros(nloc, device=dev, dtype=torch.float64)

    inv_w = 1.0 / X0_M[THREAT_MAT]
    inv_s = 1.0 / X0_M[CONFUSER_MAT]

    s_mid = ((torch.arange(nstep, device=dev, dtype=torch.float64) + 0.5)
             / nstep)

    n = obs_all["entry"].shape[0]
    for lo in range(0, n, track_chunk):
        hi = min(lo + track_chunk, n)
        obs = {k: v[lo:hi] for k, v in obs_all.items()}
        e, d, t_exit = obs["entry"], obs["d"], obs["t_exit"]

        pts = e[:, None, :] + (t_exit[:, None] * s_mid)[..., None] * d[:, None, :]
        base_inv = inv_X0_at(pts, base_manifest, dev)

        ds = (t_exit / nstep)[:, None]
        lever = t_exit[:, None] * (1.0 - s_mid)
        w0 = ds
        w1 = ds * lever
        w2 = ds * lever * lever

        baseJ0 = (base_inv * w0).sum(1)
        baseJ1 = (base_inv * w1).sum(1)
        baseJ2 = (base_inv * w2).sum(1)

        ll_none += marginal_event_logpdf(
            baseJ0, baseJ1, baseJ2, obs, pgrid, logw
        ).sum()

        for l0 in range(0, nloc, loc_chunk):
            l1 = min(l0 + loc_chunk, nloc)
            cg = centers[l0:l1]
            inside = ((pts[None, ...] - cg[:, None, None, :]).abs() < half).all(-1)

            def target_moments(inv_target):
                delta = (inv_target - base_inv)[None, :, :] * inside
                J0 = baseJ0[None, :] + (delta * w0[None, :, :]).sum(2)
                J1 = baseJ1[None, :] + (delta * w1[None, :, :]).sum(2)
                J2 = baseJ2[None, :] + (delta * w2[None, :, :]).sum(2)
                return J0, J1, J2

            J0, J1, J2 = target_moments(inv_w)
            ll_w[l0:l1] += marginal_event_logpdf(
                J0, J1, J2, obs, pgrid, logw
            ).sum(1)

            J0, J1, J2 = target_moments(inv_s)
            ll_s[l0:l1] += marginal_event_logpdf(
                J0, J1, J2, obs, pgrid, logw
            ).sum(1)

        del pts, base_inv, w0, w1, w2

    return dict(no_target=ll_none, tungsten=ll_w, steel=ll_s)


def schedule_bank(schedule_events, base_manifest, centers, target_size,
                  nstep, nq, dev, track_chunk, loc_chunk):
    """Combine independent views by summing their log likelihoods."""
    total = None
    for view_name, ev in schedule_events:
        if view_name == "A":
            bm, cc = base_manifest, centers
        elif view_name == "B":
            bm, cc = rotate_manifest_y90(base_manifest), rotate_vec_y90(centers)
        else:
            raise ValueError(view_name)

        bank = likelihood_bank(
            ev, bm, cc, target_size, nstep, nq, dev,
            track_chunk=track_chunk, loc_chunk=loc_chunk,
        )
        if total is None:
            total = bank
        else:
            total = {
                "no_target": total["no_target"] + bank["no_target"],
                "tungsten": total["tungsten"] + bank["tungsten"],
                "steel": total["steel"] + bank["steel"],
            }
    return total


# ---------------------------------------------------------------------------
# Evidence calculations
# ---------------------------------------------------------------------------

def logsumexp_list(xs):
    return torch.logsumexp(torch.stack(xs), dim=0)


def scene_scores_from_banks(banks, probe_loc_idx: int):
    """
    One bank => known-cargo Phase 3A.
    Multiple banks => unknown-cargo Phase 3B.
    """
    C = len(banks)
    L = int(banks[0]["tungsten"].numel())
    logC, logL = math.log(C), math.log(L)

    h1_terms = []
    for b in banks:
        h1_terms.extend((b["tungsten"] - logC - logL).unbind())
    log_h1 = logsumexp_list(h1_terms)

    h0_terms = []
    for b in banks:
        h0_terms.append(b["no_target"] + math.log(0.5) - logC)
        h0_terms.extend((b["steel"] + math.log(0.5) - logC - logL).unbind())
    log_h0 = logsumexp_list(h0_terms)
    blind = log_h1 - log_h0

    # Location-revealed diagnostic. Probe location is defined for every scene.
    h1r = [b["tungsten"][probe_loc_idx] - logC for b in banks]
    log_h1r = logsumexp_list(h1r)

    h0r = []
    for b in banks:
        h0r.append(b["no_target"] + math.log(0.5) - logC)
        h0r.append(b["steel"][probe_loc_idx] + math.log(0.5) - logC)
    log_h0r = logsumexp_list(h0r)
    revealed = log_h1r - log_h0r

    return float(blind), float(revealed)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_specs(n, label, stream_seed, n_cargo, nloc, rid_offset):
    rng = np.random.default_rng(stream_seed)
    specs = []
    for i in range(n):
        cargo_idx = int(rng.integers(0, n_cargo))
        loc_idx = int(rng.integers(0, nloc))
        if label == 1:
            kind = "threat"
        else:
            kind = "steel" if rng.random() < 0.5 else "none"
        specs.append(SceneSpec(
            label=label,
            cargo_idx=cargo_idx,
            probe_loc_idx=loc_idx,
            benign_kind=kind,
            realization_id=rid_offset + i,
        ))
    return specs


def true_manifest(spec, bases, centers):
    base = bases[spec.cargo_idx]
    c = centers[spec.probe_loc_idx]
    if spec.label == 1:
        return insert_cube(base, THREAT_MAT, c)
    if spec.benign_kind == "steel":
        return insert_cube(base, CONFUSER_MAT, c)
    return insert_cube(base, None, c)


def seed_pair(master_seed: int, realization_id: int, schedule_id: int, view_id: int):
    x = (master_seed + realization_id * 1_000_003
         + schedule_id * 10_007 + view_id * 101)
    return int(x + 17), int(x + 53)


def simulate_schedules(spec, bases, centers, budget, nstep, dev, master_seed):
    m = true_manifest(spec, bases, centers)

    ss, sc = seed_pair(master_seed, spec.realization_id, 0, 0)
    ev_static, inc_static = simulate(m, budget, nstep, dev, ss, sc)

    nA = budget // 2
    nB = budget - nA

    ss, sc = seed_pair(master_seed, spec.realization_id, 1, 0)
    ev_A, inc_A = simulate(m, nA, nstep, dev, ss, sc)

    mr = rotate_manifest_y90(m)
    ss, sc = seed_pair(master_seed, spec.realization_id, 1, 1)
    ev_B, inc_B = simulate(mr, nB, nstep, dev, ss, sc)

    return {
        "static": ([("A", ev_static)], inc_static),
        "two_view": ([("A", ev_A), ("B", ev_B)], inc_A + inc_B),
    }


def evaluate_scene(spec, bases, centers, schedules, mode, target_size,
                   nstep, nq, dev, track_chunk, loc_chunk):
    out = {}
    if mode == "known":
        candidate_indices = [spec.cargo_idx]
    elif mode == "unknown":
        candidate_indices = list(range(len(bases)))
    else:
        raise ValueError(mode)

    for sched_name, (sched_events, incident) in schedules.items():
        banks = []
        for ci in candidate_indices:
            banks.append(schedule_bank(
                sched_events, bases[ci], centers, target_size,
                nstep, nq, dev, track_chunk, loc_chunk,
            ))
        blind, revealed = scene_scores_from_banks(banks, spec.probe_loc_idx)
        out[sched_name] = dict(blind=blind, revealed=revealed,
                               incident=int(incident))
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def auc_tie_correct(pos, neg):
    """Mann-Whitney AUC with exact 0.5 credit for ties."""
    p = np.asarray(pos, dtype=float)
    n = np.asarray(neg, dtype=float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).mean()
    eq = (p[:, None] == n[None, :]).mean()
    return float(gt + 0.5 * eq)


def bootstrap_auc_ci(pos, neg, nboot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = np.empty(nboot)
    for i in range(nboot):
        pp = pos[rng.integers(0, len(pos), len(pos))]
        nn = neg[rng.integers(0, len(neg), len(neg))]
        vals[i] = auc_tie_correct(pp, nn)
    return tuple(np.quantile(vals, [0.025, 0.975]))


def bootstrap_auc_delta_ci(pos_a, neg_a, pos_b, neg_b,
                           nboot=2000, seed=24680):
    """Paired bootstrap by scene index: B - A."""
    rng = np.random.default_rng(seed)
    pa, na = np.asarray(pos_a), np.asarray(neg_a)
    pb, nb = np.asarray(pos_b), np.asarray(neg_b)
    assert len(pa) == len(pb) and len(na) == len(nb)
    vals = np.empty(nboot)
    for i in range(nboot):
        ip = rng.integers(0, len(pa), len(pa))
        inn = rng.integers(0, len(na), len(na))
        vals[i] = (auc_tie_correct(pb[ip], nb[inn])
                   - auc_tie_correct(pa[ip], na[inn]))
    return tuple(np.quantile(vals, [0.025, 0.975]))


def threshold_at_fpr(cal_neg, fpr=0.05):
    cal_neg = np.asarray(cal_neg, dtype=float)
    try:
        return float(np.quantile(cal_neg, 1.0 - fpr, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(cal_neg, 1.0 - fpr, interpolation="higher"))


def operating_point(cal_neg, test_pos, test_neg, fpr=0.05):
    th = threshold_at_fpr(cal_neg, fpr)
    sens = float((np.asarray(test_pos) >= th).mean())
    realized_fpr = float((np.asarray(test_neg) >= th).mean())
    return th, sens, realized_fpr


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path):
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["known", "unknown"], default="known")
    ap.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    ap.add_argument("--n-cal-benign", type=int, default=200)
    ap.add_argument("--n-test-benign", type=int, default=200)
    ap.add_argument("--n-test-threat", type=int, default=200)
    ap.add_argument("--nstep", type=int, default=NSTEP)
    ap.add_argument("--quadrature", type=int, default=DEFAULT_QUAD)
    ap.add_argument("--master-seed", type=int, default=9_050_026)
    ap.add_argument("--track-chunk", type=int, default=2500)
    ap.add_argument("--loc-chunk", type=int, default=6)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # NON-SCIENTIFIC code-path validation only.
        args.budget = 500
        args.n_cal_benign = 4
        args.n_test_benign = 4
        args.n_test_threat = 4
        clutter_seeds = CLUTTER_SEEDS[:2]
        centers = frozen_locations()[[13, 4, 10, 16, 22]]
        print("WARNING: --smoke mode. Results are NON-SCIENTIFIC.")
    else:
        clutter_seeds = CLUTTER_SEEDS
        centers = frozen_locations()

    dev = torch.device(args.device)
    print(f"device: {dev}  torch {torch.__version__}")
    print(f"mode: {args.mode}")
    print(f"budget: {args.budget:,} accepted tracks/schedule")
    print(f"cargo states: {len(clutter_seeds)}")
    print(f"locations: {len(centers)}")
    print(f"momentum quadrature: {args.quadrature}")
    print(f"nstep: {args.nstep}")

    if not args.smoke:
        deviations = []
        if args.budget != PRIMARY_BUDGET:
            deviations.append(f"budget={args.budget} (protocol {PRIMARY_BUDGET})")
        if args.nstep != NSTEP:
            deviations.append(f"nstep={args.nstep} (protocol {NSTEP})")
        if (args.n_cal_benign, args.n_test_benign, args.n_test_threat) != (200, 200, 200):
            deviations.append("sample sizes differ from protocol 200/200/200")
        if deviations:
            print("\nPROTOCOL DEVIATION:")
            for d in deviations:
                print(f"  - {d}")
            print("These results must not be labeled the frozen primary gate.\n")

    bases = [base_manifest_from_seed(s) for s in clutter_seeds]

    cal_specs = make_specs(args.n_cal_benign, 0, args.master_seed + 100,
                           len(bases), len(centers), 1_000_000)
    neg_specs = make_specs(args.n_test_benign, 0, args.master_seed + 200,
                           len(bases), len(centers), 2_000_000)
    pos_specs = make_specs(args.n_test_threat, 1, args.master_seed + 300,
                           len(bases), len(centers), 3_000_000)

    all_sets = [("cal_benign", cal_specs),
                ("test_benign", neg_specs),
                ("test_threat", pos_specs)]

    scores = {
        split: {"static_blind": [], "static_revealed": [],
                "static_incident": [], "two_blind": [],
                "two_revealed": [], "two_incident": []}
        for split, _ in all_sets
    }

    total = sum(len(x) for _, x in all_sets)
    done, t0 = 0, time.time()

    for split, specs in all_sets:
        for spec in specs:
            schedules = simulate_schedules(spec, bases, centers, args.budget,
                                           args.nstep, dev, args.master_seed)
            res = evaluate_scene(spec, bases, centers, schedules, args.mode,
                                 TARGET_SIZE_M, args.nstep, args.quadrature,
                                 dev, args.track_chunk, args.loc_chunk)

            scores[split]["static_blind"].append(res["static"]["blind"])
            scores[split]["static_revealed"].append(res["static"]["revealed"])
            scores[split]["static_incident"].append(res["static"]["incident"])
            scores[split]["two_blind"].append(res["two_view"]["blind"])
            scores[split]["two_revealed"].append(res["two_view"]["revealed"])
            scores[split]["two_incident"].append(res["two_view"]["incident"])

            done += 1
            if done % max(1, min(10, total)) == 0 or done == total:
                print(f"  {done}/{total}  {time.time()-t0:8.1f}s")

    cal, neg, pos = scores["cal_benign"], scores["test_benign"], scores["test_threat"]

    print("\nBLIND IDEAL OBSERVER")
    print(f"{'schedule':>10} {'AUC':>8} {'95% CI':>19} "
          f"{'sens@5%FPR':>12} {'test FPR':>10}")
    blind_results = {}
    for name, key in [("static", "static_blind"), ("two_view", "two_blind")]:
        auc = auc_tie_correct(pos[key], neg[key])
        ci = bootstrap_auc_ci(pos[key], neg[key])
        th, sens, rfpr = operating_point(cal[key], pos[key], neg[key])
        blind_results[name] = dict(auc=auc, ci=ci, threshold=th,
                                   sensitivity=sens, fpr=rfpr)
        print(f"{name:>10} {auc:8.3f} [{ci[0]:.3f},{ci[1]:.3f}] "
              f"{sens:12.3f} {rfpr:10.3f}")

    delta = blind_results["two_view"]["auc"] - blind_results["static"]["auc"]
    dci = bootstrap_auc_delta_ci(pos["static_blind"], neg["static_blind"],
                                 pos["two_blind"], neg["two_blind"])
    print(f"\ntwo_view - static AUC = {delta:+.3f} "
          f"95% CI [{dci[0]:+.3f},{dci[1]:+.3f}]")

    print("\nLOCATION-REVEALED DIAGNOSTIC")
    print(f"{'schedule':>10} {'AUC':>8} {'95% CI':>19}")
    revealed_results = {}
    for name, key in [("static", "static_revealed"),
                      ("two_view", "two_revealed")]:
        auc = auc_tie_correct(pos[key], neg[key])
        ci = bootstrap_auc_ci(pos[key], neg[key], seed=54321)
        revealed_results[name] = dict(auc=auc, ci=ci)
        print(f"{name:>10} {auc:8.3f} [{ci[0]:.3f},{ci[1]:.3f}]")

    print("\nINCIDENT / ACCEPTED DIAGNOSTIC")
    for sched, key in [("static", "static_incident"),
                       ("two_view", "two_incident")]:
        vals = np.asarray(scores["test_benign"][key]
                          + scores["test_threat"][key], dtype=float)
        print(f"{sched:>10}: incident median {np.median(vals):,.0f}, "
              f"accepted {args.budget:,}, "
              f"acceptance~{args.budget/np.median(vals):.3f}")

    if not args.smoke:
        s_auc = blind_results["static"]["auc"]
        t_auc = blind_results["two_view"]["auc"]
        t_sens = blind_results["two_view"]["sensitivity"]
        s_sens = blind_results["static"]["sensitivity"]

        print("\nFROZEN GATE INTERPRETATION")
        if s_auc >= 0.90:
            print("Gate A candidate: static raw measurements already support "
                  "strong blind discrimination. Next branch = better STATIC "
                  "inference/reconstruction; do not pursue active geometry yet.")
        elif (t_auc >= 0.90 and delta >= 0.10 and dci[0] > 0.0
              and t_sens > s_sens):
            print("Gate B candidate: predetermined complementary geometry adds "
                  "decisive information. Next branch = acquisition-geometry "
                  "research (not yet adaptive control).")
        elif s_auc < 0.80 and t_auc < 0.80:
            print("Gate C candidate: even the favorable declared-model ideal "
                  "observer is weak at this budget. Stop this target-detection "
                  "concept rather than adding a more elaborate reconstruction.")
        else:
            print("INCONCLUSIVE region. Do not manufacture a new detector tweak "
                  "from this result.")

    metadata = dict(
        mode=args.mode, smoke=bool(args.smoke), budget=args.budget,
        nstep=args.nstep, quadrature=args.quadrature,
        n_cargo=len(bases), n_locations=len(centers),
        target_size_m=TARGET_SIZE_M, threat_material=THREAT_MAT,
        confuser_material=CONFUSER_MAT, clutter_seeds=list(clutter_seeds),
        grid_levels_m=list(GRID_LEVELS_M),
        n_cal_benign=args.n_cal_benign,
        n_test_benign=args.n_test_benign,
        n_test_threat=args.n_test_threat,
        master_seed=args.master_seed,
        protocol_sha256=sha256_file("PHASE3_PROTOCOL.md"),
        git_head=git_head(), blind_results=blind_results,
        revealed_results=revealed_results,
        auc_delta_two_minus_static=delta, auc_delta_ci=dci,
    )

    if args.save:
        arrays = {"metadata_json": np.array(json.dumps(metadata))}
        for split, dd in scores.items():
            for k, v in dd.items():
                arrays[f"{split}_{k}"] = np.asarray(v)
        np.savez_compressed(args.save, **arrays)
        print(f"\nsaved -> {args.save}")

    print("\nprotocol sha256:", metadata["protocol_sha256"])
    print("git HEAD:", metadata["git_head"])


if __name__ == "__main__":
    main()
