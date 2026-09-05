#!/usr/bin/env python3
"""
phase2c_argmax.py -- THE LAST DIAGNOSTIC before go/no-go on this branch.

Question: is the non-monotonic lambda curve (0.593 -> 0.689 -> 0.609 -> 0.592)
caused by the expanding Stage-A valid mask admitting progressively more
extreme false peaks, or is lambda simply a weak detector?

The decisive test is the ORACLE-LOCALISED AUC. Score every scene using the
maximum of the SAME filtered field restricted to the true target region. That
is not a detector -- it uses ground truth and could never be fielded -- but it
removes the search entirely. If oracle-localised AUC rises monotonically with
exposure while the global max does not, the pathology is the scan statistic
and is fixable. If oracle-localised AUC is ALSO flat or non-monotonic, then
lambda carries little exposure-dependent information and no score correction
will rescue it.

Scoring path is imported from phase2b_lambda and cross-checked against it, so
this cannot silently measure a different statistic.

    python phase2c_argmax.py --n-scenes 400 --save phase2c_dev.npz
"""
import argparse, time
import torch
import torch.nn.functional as F
from scene_physics import make_manifest, simulate, TARGET_SIZES_M
from phase2_occupancy import stage_a, roc_auc, KERNEL, MIN_KERNEL_SUPPORT, G_MIN, EPS, N_FOLDS
from phase2b_lambda import mark_fields, matched_filter_max


def filtered_field(Z, valid):
    """Same statistic as phase2b_lambda.matched_filter_max, but returns the
    whole field so we can locate the peak and restrict it."""
    x = torch.where(valid, Z, torch.zeros_like(Z))[None, None].float()
    k = torch.ones(1, 1, KERNEL, KERNEL, KERNEL, device=Z.device)
    pad = KERNEL//2
    num = F.conv3d(x, k, padding=pad)[0, 0]
    den = F.conv3d(valid[None, None].float(), k, padding=pad)[0, 0]
    return torch.where(den >= MIN_KERNEL_SUPPORT, num/den.clamp_min(1.0),
                       torch.full_like(num, float("-inf")))


def q(v, ps=(0.5, 0.9, 0.99)):
    t = torch.tensor([x for x in v if x == x and abs(x) != float("inf")],
                     dtype=torch.float64)
    if t.numel() < 5:
        return [float("nan")]*len(ps)
    return [float(torch.quantile(t, p)) for p in ps]


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
    print(f"device: {dev}  torch {torch.__version__}\n")

    G = stage_a(B, args.stage_a_reps, nv, side, args.nstep, dev)
    valid = {b: (G[b] >= G_MIN) for b in B}
    vox = side/nv
    cc = torch.arange(nv, device=dev, dtype=torch.float64)*vox - side/2 + vox/2
    XX, YY, ZZ = torch.meshgrid(cc, cc, cc, indexing="ij")
    COORD = torch.stack([XX, YY, ZZ], -1)

    print(f"\n{'budget':>9} {'valid voxels':>13} {'valid frac':>11}")
    for b in B:
        print(f"{b:9,} {int(valid[b].sum()):13,} {float(valid[b].double().mean()):11.3f}")

    print(f"\npass 1: {args.n_scenes} scenes")
    Sf = {b: [] for b in B}; Cf = {b: [] for b in B}
    labels, folds, tmasks = [], [], []
    t0 = time.time()
    for si in range(args.n_scenes):
        m = make_manifest(si, side)
        ev, _ = simulate(m, max(B), args.nstep, dev, 1000, 2000)
        from phase2_occupancy import poca_points
        pt, ok, th = poca_points(ev)
        labels.append(int(m["present"])); folds.append(si % N_FOLDS)
        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double(); h = m["target"]["h"].to(dev).double()
        else:
            # size-matched probe region so the oracle max is taken over a
            # comparable volume on both sides. Dedicated RNG stream: this is
            # a diagnostic construct and must not perturb the physics.
            gp = torch.Generator().manual_seed(770_000 + si)
            sz = TARGET_SIZES_M[int(torch.rand(1, generator=gp).item()
                                    * len(TARGET_SIZES_M))]
            lim = side/2 - sz/2 - 0.02
            c = (torch.rand(3, generator=gp)*(2*lim) - lim).to(dev).double()
            h = torch.full((3,), sz/2).to(dev).double()
        tmasks.append(((COORD - c).abs() < h).all(-1))
        for b in B:
            S, C, _, _, _ = mark_fields(pt[:b], ok[:b], th[:b], nv, side)
            Sf[b].append(S.float().cpu()); Cf[b].append(C.float().cpu())
        if (si+1) % 50 == 0:
            print(f"  {si+1}/{args.n_scenes}  {time.time()-t0:6.1f}s")

    lab = torch.tensor(labels); fold = torch.tensor(folds)
    print(f"\nscenes: {int(lab.sum())} threat / {int((1-lab).sum())} benign")

    gmax = {b: [0.0]*args.n_scenes for b in B}      # global max (the detector)
    omax = {b: [float('nan')]*args.n_scenes for b in B}  # oracle-localised
    aloc = {b: [None]*args.n_scenes for b in B}     # argmax coordinates
    aG   = {b: [float('nan')]*args.n_scenes for b in B}
    hit  = {b: [False]*args.n_scenes for b in B}
    mism = 0

    for b in B:
        S = torch.stack(Sf[b]).to(dev).double(); C = torch.stack(Cf[b]).to(dev).double()
        for k in range(N_FOLDS):
            trn = ((lab == 0) & (fold != k)).to(dev)
            mu_p = float(S[trn].sum()/C[trn].sum().clamp_min(1.0))
            MU = (S + args.alpha*mu_p)/(C + args.alpha)
            mu0 = MU[trn].mean(0); sd0 = MU[trn].std(0).clamp_min(EPS)
            for si in torch.nonzero(fold == k).flatten().tolist():
                Z = (MU[si]-mu0)/sd0
                fld = filtered_field(Z, valid[b])
                ref = matched_filter_max(Z, valid[b])
                if not torch.isfinite(fld).any():
                    continue
                mx = float(fld.max())
                if ref is None or abs(mx-ref) > 1e-4:
                    mism += 1
                gmax[b][si] = mx
                fl = int(torch.argmax(fld))
                i = (fl//(nv*nv), (fl//nv) % nv, fl % nv)
                loc = COORD[i[0], i[1], i[2]]
                aloc[b][si] = [float(x) for x in loc]
                aG[b][si] = float(G[b][i[0], i[1], i[2]])
                tm = tmasks[si]
                if tm is not None:
                    tf = torch.where(tm, fld, torch.full_like(fld, float("-inf")))
                    if torch.isfinite(tf).any():
                        omax[b][si] = float(tf.max())
                    hit[b][si] = bool(tm[i[0], i[1], i[2]])
            del MU, mu0, sd0
        del S, C
    print(f"cross-check against phase2b scoring path: {mism} mismatches "
          f"(must be 0)\n")

    edge = lambda p: min(side/2 - abs(p[0]), side/2 - abs(p[1]), side/2 - abs(p[2]))
    T = [i for i in range(args.n_scenes) if labels[i] == 1]
    N = [i for i in range(args.n_scenes) if labels[i] == 0]

    pv = torch.tensor([float(t.double().sum()) for t in tmasks if t is not None])
    pvT = torch.tensor([float(tmasks[i].double().sum()) for i in T])
    pvN = torch.tensor([float(tmasks[i].double().sum()) for i in N])
    print(f"\nprobe-region volume (voxels): threat median {float(pvT.median()):.0f}, "
          f"benign median {float(pvN.median()):.0f}  <- must be comparable")
    print(f"\n{'budget':>9} {'AUC glob':>9} {'AUC oracle':>11} {'argmax-in-probe':>16} "
          f"{'G@argmax':>10} {'edge dist':>10}")
    for b in B:
        ao = float("nan")
        oo = [omax[b][i] for i in range(args.n_scenes)
              if omax[b][i] == omax[b][i]]
        if len(oo) > 20:
            # probe-region max on BOTH sides: comparable search volumes
            lo = min(oo) - 1.0
            sc = [omax[b][i] if omax[b][i] == omax[b][i] else lo
                  for i in range(args.n_scenes)]
            ao = roc_auc(sc, labels)
        ag = torch.tensor([aG[b][i] for i in range(args.n_scenes)
                           if aG[b][i] == aG[b][i]])
        ed = torch.tensor([edge(aloc[b][i]) for i in range(args.n_scenes)
                           if aloc[b][i]])
        print(f"{b:9,} {roc_auc([gmax[b][i] for i in range(args.n_scenes)], labels):9.3f} "
              f"{ao:11.3f} {sum(hit[b][i] for i in T)/max(len(T),1):16.3f} "
              f"{float(ag.median()):10.3f} {float(ed.median()):10.3f}")

    print(f"\nscore distribution by budget (the empirical scan null is the "
          f"benign row)")
    print(f"{'budget':>9} {'benign p50':>11} {'p90':>7} {'p99':>7} | "
          f"{'threat p50':>11} {'p90':>7} {'p99':>7}")
    for b in B:
        bn = q([gmax[b][i] for i in N]); tr = q([gmax[b][i] for i in T])
        print(f"{b:9,} {bn[0]:11.3f} {bn[1]:7.3f} {bn[2]:7.3f} | "
              f"{tr[0]:11.3f} {tr[1]:7.3f} {tr[2]:7.3f}")

    print(f"\nargmax migration between consecutive budgets (median, metres)")
    for i in range(len(B)-1):
        d = [sum((a-c)**2 for a, c in zip(aloc[B[i]][s], aloc[B[i+1]][s]))**0.5
             for s in range(args.n_scenes)
             if aloc[B[i]][s] and aloc[B[i+1]][s]]
        t = torch.tensor(d)
        print(f"  {B[i]:>8,} -> {B[i+1]:>8,}   all {float(t.median()):.3f}   "
              f"(uncorrelated argmaxes would give ~0.66)")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save,
            labels=np.array(labels), budgets=np.array(B),
            gmax=np.array([[gmax[b][i] for b in B] for i in range(args.n_scenes)]),
            omax=np.array([[omax[b][i] for b in B] for i in range(args.n_scenes)]),
            aG=np.array([[aG[b][i] for b in B] for i in range(args.n_scenes)]),
            hit=np.array([[hit[b][i] for b in B] for i in range(args.n_scenes)]),
            aloc=np.array([[aloc[b][i] or [np.nan]*3 for b in B]
                           for i in range(args.n_scenes)]))
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# READ THE ORACLE COLUMN FIRST. Everything else is supporting detail.
#
#   oracle rises monotonically, global does not
#       -> scan-statistic pathology. The signal is there; the search is
#          destroying it. ONE principled correction is warranted, defined by
#          this diagnosis (e.g. a search-volume-invariant null calibrated from
#          benign scenes at each budget), then rerun DEV untouched.
#
#   oracle is ALSO flat or non-monotonic
#       -> lambda carries little exposure-dependent information under this
#          reconstruction. No score correction fixes that. STOP.
#
# The oracle score uses the TRUE target location. It is not a detector and
# must never be reported as one. Its only job is to separate "weak signal"
# from "signal destroyed by searching".
#
# G_T caveat, which applies to the earlier 0.711 vs 0.473 result too: G_T is
# computed from the known target position. It establishes that regions poorly
# covered by the instrument are much harder to detect. It does NOT establish
# that a policy could recognise which view would improve coverage -- that
# needs an observable proxy such as the reconstruction's own coverage map,
# not ground truth. Physical motivation for active view selection, not
# evidence that it would work.
