#!/usr/bin/env python3
"""
phase2d_calibrated.py -- THE one permitted correction. Then the stop rule.

DIAGNOSIS (phase2c): the global lambda detector fails because voxel-specific
null tails are not comparable, so max() systematically selects the
worst-calibrated location. Evidence: G@argmax 0.231 -> 0.162, edge distance
0.190 -> 0.010 m, benign p99 +0.63 where plain multiple comparisons predicts
+0.30, argmax migration 0.62 against 0.66 for uncorrelated. Meanwhile the
ORACLE (same field, true target location) runs 0.664 -> 0.885 -> 0.966 ->
0.971. The signal is large; the search destroys it.

CORRECTION: conditionally pooled empirical tail calibration.
  Stratify voxels by Stage-A response G(v,N) and edge distance -- both
  deterministic instrument properties, no scene data. Pool the FILTERED field
  values of benign training scenes within each stratum, and convert each
  voxel to a survival p-value:

      T(v) = -log10 P0( F >= F(v) | stratum(v) )

  A T of 4 is then equally surprising at the wall and at the centre, which is
  exactly the exchangeability that max() requires.

  NOT one tail fit per voxel: ~160 benign scenes per fold gives ~16
  exceedances above the 90th percentile, and fitting 100k fragile models
  would recreate the pathology. Pooling gives ~10^6 observations per stratum,
  enough for empirical survival ranks with no parametric tail at all.

  Calibration order matters: filter FIRST, then calibrate, because the
  calibrated quantity must be the quantity that is maximised.

MECHANISM TESTS (better than AUC alone -- if AUC improves but the argmax
still races to the wall, we have merely found another fitted statistic):
  G@argmax should stop collapsing. Edge distance should stop going to zero.
  Benign maxima should stabilise across budgets. Global AUC should become
  monotonic in exposure. The global-vs-oracle gap should shrink.

    python phase2d_calibrated.py --n-scenes 400 --save phase2d_dev.npz
"""
import argparse, time
import torch
import torch.nn.functional as F
from scene_physics import make_manifest, simulate, TARGET_SIZES_M
from phase2_occupancy import (stage_a, roc_auc, poca_points, KERNEL,
                              MIN_KERNEL_SUPPORT, G_MIN, EPS, N_FOLDS)
from phase2b_lambda import mark_fields

N_G_BINS, N_EDGE_BINS = 10, 3


def filtered_field(Z, valid):
    x = torch.where(valid, Z, torch.zeros_like(Z))[None, None].float()
    k = torch.ones(1, 1, KERNEL, KERNEL, KERNEL, device=Z.device)
    p = KERNEL//2
    num = F.conv3d(x, k, padding=p)[0, 0]
    den = F.conv3d(valid[None, None].float(), k, padding=p)[0, 0]
    return torch.where(den >= MIN_KERNEL_SUPPORT, num/den.clamp_min(1.0),
                       torch.full_like(num, float("-inf")))


def make_strata(Gb, valid, side, nvox, dev):
    """Deterministic instrument properties only. No scene data touches this."""
    vox = side/nvox
    cc = torch.arange(nvox, device=dev, dtype=torch.float64)*vox - side/2 + vox/2
    X, Y, Z_ = torch.meshgrid(cc, cc, cc, indexing="ij")
    edge = torch.minimum(torch.minimum(side/2 - X.abs(), side/2 - Y.abs()),
                         side/2 - Z_.abs())
    gv = Gb[valid]
    gq = torch.quantile(gv, torch.linspace(0, 1, N_G_BINS+1, device=dev,
                                           dtype=torch.float64)[1:-1])
    ev = edge[valid]
    eq = torch.quantile(ev, torch.linspace(0, 1, N_EDGE_BINS+1, device=dev,
                                           dtype=torch.float64)[1:-1])
    gi = torch.bucketize(Gb, gq)
    ei = torch.bucketize(edge, eq)
    strat = (gi*N_EDGE_BINS + ei)
    strat = torch.where(valid, strat, torch.full_like(strat, -1))
    return strat, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--nvox", type=int, default=50)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[25_000, 50_000, 100_000, 250_000])
    ap.add_argument("--stage-a-reps", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device); B = sorted(args.budgets)
    nv, side = args.nvox, args.side
    print(f"device: {dev}  torch {torch.__version__}")
    print(f"strata: {N_G_BINS} G-bins x {N_EDGE_BINS} edge-bins = "
          f"{N_G_BINS*N_EDGE_BINS}, from instrument response only\n")

    G = stage_a(B, args.stage_a_reps, nv, side, args.nstep, dev)
    valid = {b: (G[b] >= G_MIN) for b in B}
    strat = {}; edged = None
    for b in B:
        strat[b], edged = make_strata(G[b], valid[b], side, nv, dev)

    vox = side/nv
    cc = torch.arange(nv, device=dev, dtype=torch.float64)*vox - side/2 + vox/2
    XX, YY, ZZ = torch.meshgrid(cc, cc, cc, indexing="ij")
    COORD = torch.stack([XX, YY, ZZ], -1)

    print(f"\npass 1: {args.n_scenes} scenes")
    Sf = {b: [] for b in B}; Cf = {b: [] for b in B}
    labels, folds, probes = [], [], []
    t0 = time.time()
    for si in range(args.n_scenes):
        m = make_manifest(si, side)
        ev, _ = simulate(m, max(B), args.nstep, dev, 1000, 2000)
        pt, ok, th = poca_points(ev)
        labels.append(int(m["present"])); folds.append(si % N_FOLDS)
        # probe region: true target when present, else a random box of the
        # SAME size this scene already drew -> matched by construction
        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double(); h = m["target"]["h"].to(dev).double()
        else:
            gp = torch.Generator().manual_seed(770_000 + si)
            sz = m["target_size"]
            lim = side/2 - sz/2 - 0.02
            c = (torch.rand(3, generator=gp)*(2*lim) - lim).to(dev).double()
            h = torch.full((3,), sz/2).to(dev).double()
        probes.append(((COORD - c).abs() < h).all(-1))
        for b in B:
            S, C, _, _, _ = mark_fields(pt[:b], ok[:b], th[:b], nv, side)
            Sf[b].append(S.float().cpu()); Cf[b].append(C.float().cpu())
        if (si+1) % 50 == 0:
            print(f"  {si+1}/{args.n_scenes}  {time.time()-t0:6.1f}s")

    lab = torch.tensor(labels); fold = torch.tensor(folds)
    NS = args.n_scenes
    T_idx = [i for i in range(NS) if labels[i] == 1]
    print(f"\nscenes: {len(T_idx)} threat / {NS-len(T_idx)} benign")
    pT = torch.tensor([float(probes[i].double().sum()) for i in T_idx])
    pN = torch.tensor([float(probes[i].double().sum()) for i in range(NS)
                       if labels[i] == 0])
    print(f"probe volume (voxels): threat median {float(pT.median()):.0f}, "
          f"benign median {float(pN.median()):.0f}")

    raw = {b: [0.0]*NS for b in B}; cal = {b: [0.0]*NS for b in B}
    orc = {b: [0.0]*NS for b in B}; aG = {b: [0.0]*NS for b in B}
    aE = {b: [0.0]*NS for b in B}; sat = {b: 0 for b in B}

    for b in B:
        Sa = torch.stack(Sf[b]).to(dev).double()
        Ca = torch.stack(Cf[b]).to(dev).double()
        st = strat[b]; nstr = int(N_G_BINS*N_EDGE_BINS)
        for k in range(N_FOLDS):
            trn = torch.nonzero(((lab == 0) & (fold != k))).flatten().tolist()
            tst = torch.nonzero(fold == k).flatten().tolist()
            mu_p = float(Sa[trn].sum()/Ca[trn].sum().clamp_min(1.0))
            MUt = (Sa[trn] + args.alpha*mu_p)/(Ca[trn] + args.alpha)
            mu0 = MUt.mean(0); sd0 = MUt.std(0).clamp_min(EPS)
            # pool filtered-field values of TRAINING benign scenes by stratum
            pools = [[] for _ in range(nstr)]
            for j in range(len(trn)):
                Ff = filtered_field((MUt[j]-mu0)/sd0, valid[b])
                fin = torch.isfinite(Ff)
                for s in range(nstr):
                    msk = fin & (st == s)
                    if msk.any():
                        pools[s].append(Ff[msk])
            sortd = [torch.sort(torch.cat(p)).values if p else None
                     for p in pools]
            del MUt
            for si in tst:
                MU = (Sa[si] + args.alpha*mu_p)/(Ca[si] + args.alpha)
                Ff = filtered_field((MU-mu0)/sd0, valid[b])
                fin = torch.isfinite(Ff)
                if not fin.any():
                    continue
                raw[b][si] = float(Ff[fin].max())
                Tf = torch.full(
                    Ff.shape,
                    float("-inf"),
                    device=Ff.device,
                    dtype=torch.float64,
                )
                for s in range(nstr):
                    msk = fin & (st == s)
                    if not msk.any() or sortd[s] is None:
                        continue
                    N0 = sortd[s].numel()
                    idx = torch.searchsorted(sortd[s], Ff[msk].contiguous())
                    surv = ((N0 - idx).double()/N0).clamp_min(1.0/(N0+1))
                    Tf[msk] = -torch.log10(surv)
                cal_fin = torch.isfinite(Tf)
                if not cal_fin.any():
                    # No calibrated voxel in this scene/fold. Leave the
                    # explicit no-evidence score initialized above.
                    continue

                mx = float(Tf[cal_fin].max())
                fl = int(torch.argmax(Tf))
                i3 = (fl//(nv*nv), (fl//nv) % nv, fl % nv)

                # Saturation must be judged against the empirical pool for
                # the ACTUAL WINNING voxel's stratum, not the largest pool.
                swin = int(st[i3])
                if swin < 0 or sortd[swin] is None:
                    continue
                Nwin = sortd[swin].numel()
                cap_win = float(torch.log10(torch.tensor(
                    Nwin + 1.0, dtype=torch.float64, device=dev)))
                is_sat = mx >= cap_win - 1e-9
                if is_sat:
                    sat[b] += 1

                # Raw-field tie break only matters when this winning stratum
                # is empirically saturated.
                cal[b][si] = mx + (1e-6*raw[b][si] if is_sat else 0.0)

                aG[b][si] = float(G[b][i3]); aE[b][si] = float(edged[i3])
                pf = torch.where(probes[si], Ff, torch.full_like(Ff, float("-inf")))
                orc[b][si] = float(pf.max()) if torch.isfinite(pf).any() else float("-inf")
            del mu0, sd0, sortd, pools
        del Sa, Ca

    print(f"\n{'budget':>9} {'AUC raw':>8} {'AUC calib':>10} {'AUC oracle':>11} "
          f"{'gap':>7} {'G@amax':>8} {'edge@amax':>10} {'sat':>5}")
    for b in B:
        o = [x for x in orc[b] if x != float("-inf")]
        lo = min(o)-1.0 if o else 0.0
        oa = roc_auc([x if x != float("-inf") else lo for x in orc[b]], labels)
        ca = roc_auc(cal[b], labels)
        print(f"{b:9,} {roc_auc(raw[b], labels):8.3f} {ca:10.3f} {oa:11.3f} "
              f"{oa-ca:7.3f} {float(torch.tensor(aG[b]).median()):8.3f} "
              f"{float(torch.tensor(aE[b]).median()):10.3f} {sat[b]:5d}")

    print(f"\nbenign calibrated-score stability (should flatten across budgets)")
    print(f"{'budget':>9} {'p50':>8} {'p90':>8} {'p99':>8}")
    for b in B:
        v = torch.tensor([cal[b][i] for i in range(NS) if labels[i] == 0])
        print(f"{b:9,} {float(v.median()):8.3f} "
              f"{float(torch.quantile(v,0.9)):8.3f} "
              f"{float(torch.quantile(v,0.99)):8.3f}")

    se = (0.25/min(len(T_idx), NS-len(T_idx)))**0.5
    print(f"\nHanley-McNeil SE ~ {se:.3f}; differences below ~{2*se:.2f} are noise.")
    print("\nSTOP RULE: if AUC calib is still mainly in the 0.6s, still declines\n"
          "with exposure, or edge@amax still marches to zero, stop this branch.")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save, labels=np.array(labels),
            budgets=np.array(B),
            raw=np.array([[raw[b][i] for b in B] for i in range(NS)]),
            cal=np.array([[cal[b][i] for b in B] for i in range(NS)]),
            orc=np.array([[orc[b][i] for b in B] for i in range(NS)]),
            aG=np.array([[aG[b][i] for b in B] for i in range(NS)]),
            aE=np.array([[aE[b][i] for b in B] for i in range(NS)]))
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# This is the ONE correction permitted by the stop rule. It is defined by the
# phase2c diagnosis, not chosen from a menu. If it fails, the branch stops and
# the honest conclusion is:
#   PoCA carries strong local high-Z information (oracle AUC 0.97 at 250k),
#   but under heterogeneous unknown cargo, scene-level detection is limited by
#   spatial coverage and by extreme-value structure in the search.
#
# Strata come from G(v,N) and edge distance -- instrument properties computed
# by Stage A, containing no scene data, so stratification cannot leak labels.
# Calibration pools only BENIGN TRAINING scenes of the current fold; an
# evaluated scene never contributes to its own calibration distribution.
#
# Saturation: a scene whose max exceeds every pooled benign value is capped at
# log10(N_pool+1) and tie-broken by the raw filtered max. The `sat` column
# counts these. If it is large, the pool needs more benign scenes, not a
# parametric tail.
#
# Judge the MECHANISM columns, not only AUC. If AUC rises while edge@amax
# still collapses toward zero, the diagnosed problem was not fixed and we have
# only found another fitted statistic -- which is a stop, not a success.
