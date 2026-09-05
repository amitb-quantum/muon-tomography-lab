#!/usr/bin/env python3
"""
phase2_occupancy.py -- occupancy-ONLY detector. No lambda. No min_count.

The question, properly isolated at last:
  After removing the instrument's own PoCA occupancy pattern and ordinary
  benign-cargo structure, does residual spatial concentration of PoCA points
  contain threat information?

  C(v)                      raw PoCA counts
    -> / G(v|N)             Stage A: empty-manifest instrument response (MC)
    -> (R - mu0)/sd0        Stage B: benign-cargo null, cross-fitted
    -> matched filter        mask-normalised 3^3, validity from GEOMETRY only
    -> max                  scene score

FROZEN before any output was inspected:
  theta_min = 0. The cluttered sweep showed a 20 mrad cut discards 42% of
  events for a 7% localisation gain, so the cut is not a useful operating
  point. Angle is recorded as metadata only.
  Validity mask comes from G(v) >= G_MIN -- instrument geometry, NOT from
  per-voxel data counts. min_count previously acted as a hidden target
  selector and is deliberately absent.

  python phase2_occupancy.py --n-scenes 400 --stage-a-reps 20

This file reports lambda nowhere. That is Phase 2b.
"""
import argparse, time
import torch
import torch.nn.functional as F
from scene_physics import (make_manifest, simulate, manifest_hash, X0_M)

KERNEL = 3
MIN_KERNEL_SUPPORT = 8
G_MIN = 0.25            # voxels the instrument cannot see are excluded
EPS = 1e-9
N_FOLDS = 5
UNSCORED_SEMANTICS = "no_alarm"   # predefined; never a silent exclusion


# ------------------------------------------------------------------ recon
def poca_points(ev, n=None):
    e, di, x, do = ev["entry"], ev["d_in"], ev["exit"], ev["d_out"]
    th = ev["theta"]
    if n is not None:
        e, di, x, do, th = (t[:n] for t in (e, di, x, do, th))
    w0 = e - x
    a = (di*di).sum(1); b = (di*do).sum(1); c = (do*do).sum(1)
    dd = (di*w0).sum(1); f = (do*w0).sum(1)
    den = a*c - b*b                      # == sin^2(theta) for unit vectors
    ok = den.abs() > 1e-14
    den = torch.where(ok, den, torch.ones_like(den))
    sc = (b*f - c*dd)/den; tc = (a*f - b*dd)/den
    return 0.5*((e + sc[:, None]*di) + (x + tc[:, None]*do)), ok, th


def count_field(pt, ok, nvox, side):
    """Raw PoCA occupancy. Returns field plus in/out-of-volume provenance."""
    inside = ok & (pt.abs() < side/2).all(1)
    n_in = int(inside.sum()); n_out = int(ok.sum()) - n_in
    v = (((pt[inside] + side/2)/(side/nvox)).long()).clamp_(0, nvox-1)
    lin = (v[:, 0]*nvox + v[:, 1])*nvox + v[:, 2]
    cnt = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    cnt.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.float64))
    return cnt.reshape((nvox,)*3), n_in, n_out


# ------------------------------------------------------------------ stage A
def d4_symmetrise(g):
    """8-fold dihedral symmetry in xy. Free variance reduction: the cosmic
    flux is azimuthally symmetric and the aperture is square."""
    acc = torch.zeros_like(g)
    for t in (False, True):
        h = g.transpose(0, 1) if t else g
        for fx in (False, True):
            for fy in (False, True):
                k = h
                if fx: k = torch.flip(k, [0])
                if fy: k = torch.flip(k, [1])
                acc += k
    return acc/8.0


def stage_a(budgets, reps, nvox, side, nstep, dev, seed0=90_000):
    """Empty-manifest Monte Carlo. Calibrates the INSTRUMENT, not the cargo:
    angular acceptance, PoCA spatial bias, near-parallel degeneracy."""
    empty = make_manifest(0, side)
    empty["boxes"] = []; empty["target"] = None
    empty["present"] = False; empty["target_mat"] = None
    G = {b: torch.zeros((nvox,)*3, device=dev, dtype=torch.float64)
         for b in budgets}
    print(f"stage A: empty-manifest calibration, {reps} reps")
    for r in range(reps):
        ev, _ = simulate(empty, max(budgets), nstep, dev,
                         seed0 + 2*r, seed0 + 2*r + 1)
        pt, ok, _ = poca_points(ev)
        for b in budgets:
            c, _, _ = count_field(pt[:b], ok[:b], nvox, side)
            G[b] += c
    for b in budgets:
        G[b] = d4_symmetrise(G[b]/reps)
        sw = G[b].sum((0, 1))
        print(f"  N={b:>9,}  mean {G[b].mean():7.3f}/voxel  "
              f"visible {float((G[b] >= G_MIN).double().mean()):.3f}  "
              f"z-slice p90/p10 {torch.quantile(sw, 0.9)/torch.quantile(sw, 0.1).clamp_min(EPS):5.2f}")
    return G


# ------------------------------------------------------------------ score
def matched_filter_max(Z, valid):
    """Mask-normalised. Sparsity cannot leak into the detector."""
    x = (Z*valid)[None, None].float()
    k = torch.ones(1, 1, KERNEL, KERNEL, KERNEL, device=Z.device)
    pad = KERNEL//2
    num = F.conv3d(x, k, padding=pad)[0, 0]
    den = F.conv3d(valid[None, None].float(), k, padding=pad)[0, 0]
    field = torch.where(den >= MIN_KERNEL_SUPPORT, num/den.clamp_min(1.0),
                        torch.full_like(num, float("-inf")))
    if not torch.isfinite(field).any():
        return None
    return float(field.max())


def roc_auc(scores, labels):
    s = torch.tensor(scores, dtype=torch.float64)
    l = torch.tensor(labels, dtype=torch.float64)
    if l.sum() < 2 or (1-l).sum() < 2:
        return float("nan")
    r = torch.argsort(torch.argsort(s)).double() + 1
    npos, nneg = int(l.sum()), int((1-l).sum())
    return float((r[l == 1].sum() - npos*(npos+1)/2)/(npos*nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--nvox", type=int, default=50)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[25_000, 50_000, 100_000, 250_000])
    ap.add_argument("--stage-a-reps", type=int, default=20)
    ap.add_argument("--bank-offset", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    B = sorted(args.budgets)
    print(f"device: {dev}  torch {torch.__version__}")
    print(f"theta_min = 0 (frozen). validity from G>= {G_MIN}, "
          f"NOT from data counts.\n")

    G = stage_a(B, args.stage_a_reps, args.nvox, args.side, args.nstep, dev)
    valid = {b: (G[b] >= G_MIN) for b in B}

    # ---- pass 1: per-scene R fields (kept on CPU, fp32) ----
    print(f"\npass 1: {args.n_scenes} scenes")
    R = {b: [] for b in B}
    labels, folds, prov = [], [], []
    t0 = time.time()
    for si in range(args.n_scenes):
        m = make_manifest(args.bank_offset + si, args.side)
        ev, n_inc = simulate(m, max(B), args.nstep, dev, 1000, 2000)
        pt, ok, th = poca_points(ev)
        labels.append(int(m["present"])); folds.append(si % N_FOLDS)
        pr = dict(scene=si, n_incident=n_inc, n_accepted=int(ok.numel()),
                  present=int(m["present"]), mat=m["target_mat"] or "none",
                  size=m["target_size"], fill=m["fill"],
                  mhash=manifest_hash(m), median_theta_mrad=float(th.median()*1e3))
        for b in B:
            c, n_in, n_out = count_field(pt[:b], ok[:b], args.nvox, args.side)
            pr[f"in_{b}"], pr[f"out_{b}"] = n_in, n_out
            R[b].append(((c/(G[b] + EPS))).float().cpu())
        prov.append(pr)
        if (si+1) % 50 == 0:
            print(f"  {si+1}/{args.n_scenes}  {time.time()-t0:6.1f}s")

    lab = torch.tensor(labels); fold = torch.tensor(folds)
    print(f"\nscenes {args.n_scenes}: {int(lab.sum())} threat / "
          f"{int((1-lab).sum())} benign")
    ofr = [sum(p[f'out_{b}'] for p in prov)/max(sum(p[f'in_{b}']+p[f'out_{b}']
           for p in prov), 1) for b in B]
    print("outside-volume PoCA fraction by budget: " +
          "  ".join(f"{b//1000}k={f:.3f}" for b, f in zip(B, ofr)))

    # is outside_fraction itself label-correlated? diagnostic ONLY, not used
    for b in B:
        of = [p[f"out_{b}"]/max(p[f"in_{b}"]+p[f"out_{b}"], 1) for p in prov]
        print(f"  diagnostic AUC of outside_fraction alone at {b:>8,}: "
              f"{roc_auc(of, labels):.3f}")

    # ---- pass 2: Stage B cross-fitted benign null, then score ----
    print(f"\npass 2: stage B benign null, {N_FOLDS}-fold cross-fitting\n")
    print(f"{'budget':>9} {'AUC':>7} {'n_scored':>9} {'n_unscored':>11} "
          f"{'mu0':>7} {'sd0':>7}")
    rows = []
    for b in B:
        stack = torch.stack(R[b]).to(dev).double()        # (scenes, x, y, z)
        scores, unscored = [], 0
        for k in range(N_FOLDS):
            trn = ((lab == 0) & (fold != k)).to(dev)
            if int(trn.sum()) < 10:
                raise RuntimeError(f"fold {k}: too few benign training scenes")
            mu0 = stack[trn].mean(0)
            sd0 = stack[trn].std(0).clamp_min(EPS)
            for si in torch.nonzero(fold == k).flatten().tolist():
                Z = (stack[si] - mu0)/sd0
                s = matched_filter_max(Z, valid[b])
                if s is None:
                    unscored += 1
                    s = float("-inf") if UNSCORED_SEMANTICS == "no_alarm" else 0.0
                scores.append((si, s))
        scores.sort()
        sv = [s for _, s in scores]
        finite = [x for x in sv if x != float("-inf")]
        lo = min(finite) - 1.0 if finite else 0.0
        sv = [lo if x == float("-inf") else x for x in sv]
        auc = roc_auc(sv, labels)
        rows.append((b, auc, len(sv)-unscored, unscored))
        print(f"{b:9,} {auc:7.3f} {len(sv)-unscored:9d} {unscored:11d} "
              f"{float(mu0.mean()):7.3f} {float(sd0.mean()):7.3f}")
        del stack

    n1, n0 = int(lab.sum()), int((1-lab).sum())
    print(f"\nHanley-McNeil SE at these class sizes is roughly "
          f"{(0.25/min(n1, n0))**0.5:.3f}; treat differences below "
          f"~{2*(0.25/min(n1, n0))**0.5:.2f} as noise.")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save,
            auc=np.array([(b, a, s, u) for b, a, s, u in rows]),
            labels=np.array(labels), folds=np.array(folds),
            prov=np.array([[p["scene"], p["present"], p["size"], p["fill"],
                            p["n_accepted"], p["median_theta_mrad"]]
                           for p in prov]),
            mats=np.array([p["mat"] for p in prov]),
            mhash=np.array([p["mhash"] for p in prov]))
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# Provenance rule, honoured: no silent exclusions. Undefined scores are
# counted, reported, and given predefined semantics (no_alarm), never dropped
# from the ROC. That bug produced a spurious AUC of 1.000 earlier.
#
# outside_fraction is REPORTED as a diagnostic and is NOT part of the
# detector. If its standalone AUC is materially above 0.5, that is a finding
# to investigate on its own terms, not a feature to quietly fold in.
#
# Stage A is an instrument-response calibration, not a geometry model. It
# absorbs angular acceptance, the PoCA estimator's central spatial bias
# (measured at 7.98x z-density swing in an empty volume with a PERFECT
# detector), and near-parallel degeneracy. D4 symmetrisation in xy is applied
# because the cosmic flux is azimuthally symmetric and the aperture square;
# check the residual asymmetry if the aperture ever becomes rectangular.
#
# Cross-fitting: a benign scene is never scored against a null it helped
# build. Threat scenes use the same out-of-fold null as their fold-mates.
# After the architecture freezes, refit mu0/sd0 on ALL dev benigns once, then
# never update on cal_v2 or test.
#
# Not yet done: lambda channel (2b), fusion (2c), detector smearing (belongs
# between events and reconstruction), Geant4 cross-check, per-event
# localisation confidence weighting using OBSERVABLE quantities only.
