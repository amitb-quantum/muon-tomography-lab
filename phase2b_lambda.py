#!/usr/bin/env python3
"""
phase2b_lambda.py -- lambda-ONLY detector. Occupancy may NOT gate anything.

Marked point process factorisation:
    PoCA position = point;  log(theta^2) = mark attached to it.
    occupancy asks: is the point DENSITY unusual?          (dead: AUC ~0.52)
    lambda asks:    are the MARKS unusual, GIVEN the count?  (this file)

The old `count >= k -> take median, else discard` turned occupancy into a
hidden classifier. Replaced by shrinkage:

    mu_hat(v) = (sum_i m_i + alpha*mu_prior) / (n(v) + alpha)

Count controls UNCERTAINTY. Large count is never itself positive evidence.
A voxel with zero events is shrunk fully to the prior -- neutral, not excluded.

MARK CHOICE, frozen a priori: m = log(theta^2). For MCS, theta^2 in the plane
is ~exponential, so its raw mean is outlier-dominated -- which is exactly the
heavy tail that made max() over 57k unstable voxel medians behave as noise
earlier. log makes the mean a well-behaved estimator of log scattering
density up to an additive constant.

    python phase2b_lambda.py --n-scenes 400 --save phase2b_dev.npz
    python phase2b_lambda.py --alpha 0.5 --alpha-sweep      # DEV only
"""
import argparse, time
import torch
import torch.nn.functional as F
from scene_physics import make_manifest, simulate, manifest_hash
from phase2_occupancy import (poca_points, count_field, stage_a, roc_auc,
                              KERNEL, MIN_KERNEL_SUPPORT, G_MIN, EPS, N_FOLDS)

UNSCORED_SEMANTICS = "no_alarm"


def mark_fields(pt, ok, th, nvox, side):
    """Returns (sum of log(theta^2) per voxel, count per voxel, provenance).
    No thresholding anywhere."""
    inside = ok & (pt.abs() < side/2).all(1)
    n_in = int(inside.sum()); n_out = int(ok.sum()) - n_in
    p, t = pt[inside], th[inside]
    v = (((p + side/2)/(side/nvox)).long()).clamp_(0, nvox-1)
    lin = (v[:, 0]*nvox + v[:, 1])*nvox + v[:, 2]
    m = torch.log((t*1e3)**2 + 1e-12)          # theta in mrad, then log
    S = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    C = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    S.scatter_add_(0, lin, m)
    C.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.float64))
    sh = (nvox,)*3
    return S.reshape(sh), C.reshape(sh), n_in, n_out, float(m.mean())


def matched_filter_max(Z, valid):
    """torch.where, NOT Z*valid: multiplication evaluates 0*nan = nan and one
    nan poisons the entire convolved field, silently turning a scene into
    'unscored' and hence 'cleared'."""
    x = torch.where(valid, Z, torch.zeros_like(Z))[None, None].float()
    k = torch.ones(1, 1, KERNEL, KERNEL, KERNEL, device=Z.device)
    pad = KERNEL//2
    num = F.conv3d(x, k, padding=pad)[0, 0]
    den = F.conv3d(valid[None, None].float(), k, padding=pad)[0, 0]
    field = torch.where(den >= MIN_KERNEL_SUPPORT, num/den.clamp_min(1.0),
                        torch.full_like(num, float("-inf")))
    if not torch.isfinite(field).any():
        return None
    return float(field.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--nvox", type=int, default=50)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[25_000, 50_000, 100_000, 250_000])
    ap.add_argument("--stage-a-reps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="shrinkage strength in virtual observations")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="DEV ONLY. Never run against cal or test.")
    ap.add_argument("--bank-offset", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    B = sorted(args.budgets)
    print(f"device: {dev}  torch {torch.__version__}")
    print(f"mark = log(theta^2 in mrad^2), alpha = {args.alpha} "
          f"virtual obs, no count gate\n")

    G = stage_a(B, args.stage_a_reps, args.nvox, args.side, args.nstep, dev)
    valid = {b: (G[b] >= G_MIN) for b in B}
    for b in B:
        gv = G[b][valid[b]]
        print(f"  N={b:>9,} valid-voxel G: mean {float(gv.mean()):.3f} "
              f"min {float(gv.min()):.3f}")

    print(f"\npass 1: {args.n_scenes} scenes")
    Sf = {b: [] for b in B}; Cf = {b: [] for b in B}
    labels, folds, prov, GT = [], [], [], []
    t0 = time.time()
    for si in range(args.n_scenes):
        m = make_manifest(args.bank_offset + si, args.side)
        ev, n_inc = simulate(m, max(B), args.nstep, dev, 1000, 2000)
        pt, ok, th = poca_points(ev)
        labels.append(int(m["present"])); folds.append(si % N_FOLDS)
        # G_T: Stage-A instrument coverage OF THE TARGET REGION
        gt = {}
        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double(); h = m["target"]["h"].to(dev).double()
            vox = args.side/args.nvox
            cc = (torch.arange(args.nvox, device=dev, dtype=torch.float64)*vox
                  - args.side/2 + vox/2)
            X, Y, Z_ = torch.meshgrid(cc, cc, cc, indexing="ij")
            tmask = ((torch.stack([X, Y, Z_], -1) - c).abs() < h).all(-1)
            for b in B:
                gt[b] = float(G[b][tmask].sum()) if int(tmask.sum()) else 0.0
        GT.append(gt)
        pr = dict(scene=si, present=int(m["present"]), mat=m["target_mat"] or "none",
                  size=m["target_size"], fill=m["fill"], n_incident=n_inc,
                  mhash=manifest_hash(m))
        for b in B:
            S, C, n_in, n_out, mbar = mark_fields(pt[:b], ok[:b], th[:b],
                                                  args.nvox, args.side)
            pr[f"in_{b}"], pr[f"out_{b}"] = n_in, n_out
            Sf[b].append(S.float().cpu()); Cf[b].append(C.float().cpu())
        prov.append(pr)
        if (si+1) % 50 == 0:
            print(f"  {si+1}/{args.n_scenes}  {time.time()-t0:6.1f}s")

    lab = torch.tensor(labels); fold = torch.tensor(folds)
    print(f"\nscenes {args.n_scenes}: {int(lab.sum())} threat / "
          f"{int((1-lab).sum())} benign")

    alphas = ([0.25, 0.5, 1.0, 2.0, 5.0] if args.alpha_sweep else [args.alpha])
    if args.alpha_sweep:
        print("\n!! ALPHA SWEEP: development only. Fitting this against cal or\n"
              "   test would make the reported FPR meaningless.\n")
    results = {}
    for alpha in alphas:
        print(f"\nalpha = {alpha}")
        print(f"{'budget':>9} {'AUC':>7} {'scored':>7} {'unsc':>5} "
              f"{'AUC|GT lo':>10} {'AUC|GT hi':>10}")
        for b in B:
            S = torch.stack(Sf[b]).to(dev).double()
            C = torch.stack(Cf[b]).to(dev).double()
            scores = [0.0]*args.n_scenes; unscored = 0
            for k in range(N_FOLDS):
                trn = ((lab == 0) & (fold != k)).to(dev)
                mu_prior = float(S[trn].sum()/C[trn].sum().clamp_min(1.0))
                MU = (S + alpha*mu_prior)/(C + alpha)         # shrinkage
                mu0 = MU[trn].mean(0); sd0 = MU[trn].std(0).clamp_min(EPS)
                for si in torch.nonzero(fold == k).flatten().tolist():
                    s = matched_filter_max((MU[si]-mu0)/sd0, valid[b])
                    if s is None:
                        unscored += 1; s = float("-inf")
                    scores[si] = s
                del MU, mu0, sd0
            fin = [x for x in scores if x != float("-inf")]
            lo = (min(fin)-1.0) if fin else 0.0
            scores = [lo if x == float("-inf") else x for x in scores]
            auc = roc_auc(scores, labels)
            # stratify by target instrument coverage G_T (threat scenes only)
            tg = [(GT[i].get(b, 0.0), i) for i in range(args.n_scenes)
                  if labels[i] == 1]
            tg.sort(); half = len(tg)//2
            negs = [i for i in range(args.n_scenes) if labels[i] == 0]
            def sub(idx):
                ii = idx + negs
                return roc_auc([scores[i] for i in ii], [labels[i] for i in ii])
            alo = sub([i for _, i in tg[:half]]); ahi = sub([i for _, i in tg[half:]])
            print(f"{b:9,} {auc:7.3f} {args.n_scenes-unscored:7d} {unscored:5d} "
                  f"{alo:10.3f} {ahi:10.3f}")
            results[(alpha, b)] = (auc, unscored, alo, ahi)
            del S, C

    n1, n0 = int(lab.sum()), int((1-lab).sum())
    se = (0.25/min(n1, n0))**0.5
    print(f"\nHanley-McNeil SE ~ {se:.3f}; treat differences below "
          f"~{2*se:.2f} as noise. The G_T columns split threat scenes in half, "
          f"so their SE is ~{(0.25/min(n1//2, n0))**0.5:.3f}.")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save,
            keys=np.array([[a, b] for (a, b) in results]),
            vals=np.array([results[k] for k in results]),
            labels=np.array(labels), folds=np.array(folds),
            GT=np.array([[g.get(b, 0.0) for b in B] for g in GT]),
            mats=np.array([p["mat"] for p in prov]),
            mhash=np.array([p["mhash"] for p in prov]))
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# What is deliberately absent: min_count, any count threshold, any use of
# occupancy as a gate. Validity comes from Stage A (G >= G_MIN), which is
# instrument geometry and independent of the scene's data.
#
# G_MIN is still a chosen rule, not a physical visibility limit. A target in
# a low-G region can scatter informatively even where empty-air expected
# occupancy is below 0.25 -- refusing to score there is algorithmic blindness,
# not physical. The AUC|GT columns exist to measure that: if detection tracks
# target instrument coverage G_T, geometry coverage is part of the problem,
# not just dwell time, and active-view selection returns for a physical
# reason rather than a novelty-hunting one.
#
# alpha is frozen at 1.0. --alpha-sweep is DEV ONLY; running it against cal
# or test converts alpha into a fitted hyperparameter and voids the FPR.
#
# Carried forward, unresolved: no Geant4 cross-check; detector smearing not
# applied (belongs between events and reconstruction); per-event localisation
# confidence must use OBSERVABLE quantities only -- never true material class
# or true scatter position, which the simulator knows and the detector cannot.
