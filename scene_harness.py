#!/usr/bin/env python3
"""
scene_harness.py -- scene-level detection for muon scattering tomography.

Replaces voxel AUC (which requires knowing where the target is) with the
operational question: given an unknown container, alarm or clear?

Design:
  * Crossed variance: n_scenes x n_muon_seeds, SAME muon seed set reused
    across scenes, so scene / muon / interaction variance separate cleanly.
  * Four independent RNG streams: scene, source, scatter, detector.
  * Scene score = MASK-NORMALIZED matched filter. Invalid voxels are excluded
    from both numerator and denominator, so sparsity cannot become part of
    the detector. Frozen a priori at 3x3x3 with MIN_KERNEL_SUPPORT=8.
  * Three disjoint scene banks: dev / cal / test. Threshold is calibrated on
    cal, applied once to test. Kernel size is NOT to be swept on test.

    python scene_harness.py                      # dev bank, ~5 min
    python scene_harness.py --bank cal --save cal.npz
    python scene_harness.py --n-scenes 200 --n-seeds 5

PREREGISTERED at time of writing, before any output was inspected:
  H1  Relative scene difficulty from early exposure predicts relative
      difficulty at later exposure.  (prelim: r=0.984 over 10 scenes)
  H2  A sequential stopping rule using early evidence reduces mean exposure
      vs fixed dwell at equal scene-level TPR/FPR.
  H3  Exact material ID from scattering alone fails for W vs U (1.11x
      contrast) while threat-vs-benign succeeds. Expected NEGATIVE result.
"""
import argparse, time
import torch
import torch.nn.functional as F

X0_CM = {"air": 30400.0, "steel": 1.757, "tungsten": 0.3504, "water": 36.08,
         "lead": 0.5612, "uranium": 0.3166}
P_REF_MEV = 3000.0
FLUX_PER_CM2_MIN = 1.0

# ---- FROZEN detector statistic. Do not tune against the test bank. ----
KERNEL = 3
MIN_KERNEL_SUPPORT = 8      # of 27 voxels must be valid
MIN_COUNT = 2               # tracks per voxel to call it valid
THREAT_MATS = ["lead", "tungsten", "uranium"]
TARGET_SIZES_M = [0.06, 0.08, 0.10, 0.14]
BANK_OFFSET = {"dev": 0, "cal": 100_000, "test": 200_000}


# ------------------------------------------------------------------ scenes
def make_scene(scene_seed, nvox, side, dev, fill=0.35):
    """Random cargo + randomly placed threat / benign confuser / nothing."""
    g = torch.Generator().manual_seed(scene_seed)
    inv = torch.full((nvox,)*3, 100.0/X0_CM["air"])
    placed = 0
    while placed < fill*nvox**3:
        sz = torch.randint(4, 14, (3,), generator=g)
        o = torch.stack([torch.randint(0, nvox-int(s), (1,), generator=g)[0]
                         for s in sz])
        mat = "steel" if torch.rand(1, generator=g).item() < 0.45 else "water"
        inv[o[0]:o[0]+sz[0], o[1]:o[1]+sz[1], o[2]:o[2]+sz[2]] = 100.0/X0_CM[mat]
        placed += int(sz.prod())
    realized_fill = (inv > 1.0).double().mean().item()

    u = torch.rand(4, generator=g)
    present = u[0].item() < 0.5
    if present:
        mat = THREAT_MATS[int(u[1].item()*len(THREAT_MATS))]
    elif u[1].item() < 0.5:
        mat = "steel"                      # benign dense confuser
    else:
        mat = None                         # nothing added
    size = TARGET_SIZES_M[int(u[2].item()*len(TARGET_SIZES_M))]

    center = torch.zeros(3)
    if mat is not None:
        margin = size/2 + 0.05
        lo, hi = -side/2 + margin, side/2 - margin
        center = torch.rand(3, generator=g)*(hi-lo) + lo
        vox = side/nvox
        c = torch.arange(nvox)*vox - side/2 + vox/2
        X, Y, Z = torch.meshgrid(c, c, c, indexing="ij")
        m = (((X-center[0]).abs() < size/2) & ((Y-center[1]).abs() < size/2) &
             ((Z-center[2]).abs() < size/2))
        inv[m] = 100.0/X0_CM[mat]
    truth = dict(present=present, material=mat, size=size,
                 center=center.tolist(), fill=realized_fill)
    return inv.to(dev), truth


# ------------------------------------------------------------------ physics
def gen_muons(n, side, dev, g, th_max=torch.pi/3):
    u = torch.rand(n, device=dev, dtype=torch.float64, generator=g)
    cmax = torch.cos(torch.tensor(th_max, dtype=torch.float64))**4
    cth = (1.0 - u*(1.0-cmax))**0.25
    sth = torch.sqrt(1.0 - cth**2)
    phi = torch.rand(n, device=dev, dtype=torch.float64, generator=g)*2*torch.pi
    d = torch.stack([sth*torch.cos(phi), sth*torch.sin(phi), -cth], -1)
    entry = torch.stack([
        torch.rand(n, device=dev, dtype=torch.float64, generator=g)*side - side/2,
        torch.rand(n, device=dev, dtype=torch.float64, generator=g)*side - side/2,
        torch.full((n,), side/2, device=dev, dtype=torch.float64)], -1)
    p = 1000.0*(1.0 - torch.rand(n, device=dev, dtype=torch.float64,
                                 generator=g))**(-1/1.7)
    return entry, d, p


def transport(entry, d, p, inv_X0, side, nvox, nstep, g_scatter):
    n_inc = entry.shape[0]
    vox = side/nvox
    t_exit = (entry[:, 2] + side/2)/(-d[:, 2])
    ep = entry + t_exit[:, None]*d
    keep = (ep[:, 0].abs() < side/2) & (ep[:, 1].abs() < side/2)
    if not keep.any():
        return None, n_inc
    entry, d, p, t_exit = entry[keep], d[keep], p[keep], t_exit[keep]
    n, dev = p.shape[0], p.device

    s = (torch.arange(nstep, device=dev, dtype=torch.float64) + 0.5)/nstep
    pts = entry[:, None, :] + (t_exit[:, None]*s)[:, :, None]*d[:, None, :]
    idx = (((pts + side/2)/vox).long()).clamp_(0, nvox-1)
    inv = inv_X0[idx[..., 0], idx[..., 1], idx[..., 2]].double()
    del pts, idx

    X_step = inv*(t_exit/nstep)[:, None]
    logf = (1.0 + 0.038*torch.log(X_step.sum(1).clamp_min(1e-12))).clamp_min(0.0)
    th_step = (13.6/p)[:, None]*torch.sqrt(X_step)*logf[:, None]
    kx = torch.randn(n, nstep, device=dev, dtype=torch.float64, generator=g_scatter)*th_step
    ky = torch.randn(n, nstep, device=dev, dtype=torch.float64, generator=g_scatter)*th_step
    lever = t_exit[:, None]*(1.0-s)
    dx, dy = (kx*lever).sum(1), (ky*lever).sum(1)
    tx, ty = kx.sum(1), ky.sum(1)
    del kx, ky, th_step, X_step, inv, lever

    a = torch.zeros_like(d); a[:, 0] = 1.0
    a[d[:, 0].abs() > 0.9] = torch.tensor([0., 1., 0.], device=dev, dtype=torch.float64)
    xh = torch.cross(a, d, dim=1); xh = xh/xh.norm(dim=1, keepdim=True)
    yh = torch.cross(d, xh, dim=1)
    d_out = d + tx[:, None]*xh + ty[:, None]*yh
    d_out = d_out/d_out.norm(dim=1, keepdim=True)
    p_out = entry + t_exit[:, None]*d + dx[:, None]*xh + dy[:, None]*yh
    return (entry, d, p_out, d_out, p, torch.sqrt(tx**2+ty**2)), n_inc


def apply_detector(tr, sigma_m, z_near, z_far, g_det):
    entry, d, p_out, d_out, p, _ = tr
    def refit(pt, dv, za, zb):
        A = pt + ((za - pt[:, 2])/dv[:, 2])[:, None]*dv
        B = pt + ((zb - pt[:, 2])/dv[:, 2])[:, None]*dv
        for T in (A, B):
            T[:, :2] += torch.randn(T.shape[0], 2, device=T.device,
                                    dtype=T.dtype, generator=g_det)*sigma_m
        nd = B - A
        nd = nd*torch.sign((nd*dv).sum(1))[:, None]
        return A, nd/nd.norm(dim=1, keepdim=True)
    Ain, din = refit(entry, d, z_near, z_far)
    Aout, dout = refit(p_out, d_out, -z_near, -z_far)
    return Ain, din, Aout, dout, p, torch.arccos((din*dout).sum(1).clamp(-1, 1))


def poca(p1, d1, p2, d2):
    w0 = p1-p2
    a = (d1*d1).sum(1); b = (d1*d2).sum(1); c = (d2*d2).sum(1)
    dd = (d1*w0).sum(1); e = (d2*w0).sum(1)
    den = a*c - b*b
    ok = den.abs() > 1e-14
    den = torch.where(ok, den, torch.ones_like(den))
    sc = (b*e - c*dd)/den; tc = (a*e - b*dd)/den
    return 0.5*((p1 + sc[:, None]*d1) + (p2 + tc[:, None]*d2)), ok


def reconstruct(pt, theta, p, ok, side, nvox, use_momentum):
    vox = side/nvox
    inside = ok & (pt.abs() < side/2).all(1)
    pt, theta, p = pt[inside], theta[inside], p[inside]
    v = (((pt + side/2)/vox).long()).clamp_(0, nvox-1)
    lin = (v[:, 0]*nvox + v[:, 1])*nvox + v[:, 2]
    scale = (p/P_REF_MEV) if use_momentum else torch.ones_like(p)
    metric = (theta*scale)**2*1e6
    o1 = torch.argsort(metric)
    lin_s, m_s = lin[o1], metric[o1]
    o2 = torch.argsort(lin_s, stable=True)
    lin_s, m_s = lin_s[o2], m_s[o2]
    uniq, counts = torch.unique_consecutive(lin_s, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=counts.device),
                        counts.cumsum(0)[:-1]])
    lam = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    cnt = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    lam[uniq] = m_s[starts + counts//2]
    cnt[uniq] = counts.double()
    return lam.reshape((nvox,)*3), cnt.reshape((nvox,)*3)


# ---------------------------------------------------- FROZEN scene statistic
def scene_score(lam, cnt, side, nvox):
    """Mask-normalized matched filter. Invalid voxels excluded from BOTH
    numerator and denominator so sparsity cannot leak into the detector."""
    valid = (cnt >= MIN_COUNT).float()
    x = (lam.float()*valid)[None, None]
    k = torch.ones(1, 1, KERNEL, KERNEL, KERNEL, device=lam.device)
    pad = KERNEL//2
    num = F.conv3d(x, k, padding=pad)[0, 0]
    den = F.conv3d(valid[None, None], k, padding=pad)[0, 0]
    field = torch.where(den >= MIN_KERNEL_SUPPORT, num/den.clamp_min(1.0),
                        torch.full_like(num, float("-inf")))
    if not torch.isfinite(field).any():
        return float("nan"), [float("nan")]*3
    flat = torch.argmax(field)
    i = [int(flat // (nvox*nvox)), int((flat // nvox) % nvox), int(flat % nvox)]
    vox = side/nvox
    loc = [c*vox - side/2 + vox/2 for c in i]
    return float(field.reshape(-1)[flat]), loc


def roc_auc(scores, labels):
    s = torch.tensor([x for x, l in zip(scores, labels) if x == x])
    l = torch.tensor([l for x, l in zip(scores, labels) if x == x], dtype=torch.float64)
    if l.sum() < 2 or (1-l).sum() < 2:
        return float("nan")
    r = torch.argsort(torch.argsort(s)).double() + 1
    np_, nn = int(l.sum()), int((1-l).sum())
    return float((r[l == 1].sum() - np_*(np_+1)/2)/(np_*nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["dev", "cal", "test"], default="dev")
    ap.add_argument("--n-scenes", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--nvox", type=int, default=50)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[25_000, 50_000, 100_000, 150_000, 250_000])
    ap.add_argument("--no-momentum", action="store_true", default=True)
    ap.add_argument("--momentum", dest="no_momentum", action="store_false")
    ap.add_argument("--det-sigma", type=float, default=1.0)
    ap.add_argument("--plane-near", type=float, default=0.6)
    ap.add_argument("--plane-far", type=float, default=1.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    global MIN_COUNT
    MIN_COUNT = args.min_count

    dev = torch.device(args.device)
    if args.bank == "test":
        print("!! TEST BANK. Kernel and threshold must already be frozen. !!\n")
    print(f"bank={args.bank}  scenes={args.n_scenes}  muon_seeds={args.n_seeds}  "
          f"momentum={'NO' if args.no_momentum else 'yes'}  det={args.det_sigma}mm")
    print(f"statistic: mask-normalized {KERNEL}^3 filter, "
          f"min_support={MIN_KERNEL_SUPPORT}, min_count={MIN_COUNT}\n")

    off = BANK_OFFSET[args.bank]
    rows, t0, n_inc_tot, n_acc_tot = [], time.time(), 0, 0
    maxb = max(args.budgets)

    for si in range(args.n_scenes):
        inv_X0, truth = make_scene(off + si, args.nvox, args.side, dev)
        for mi in range(args.n_seeds):
            g_src = torch.Generator(device=dev).manual_seed(1_000 + mi)
            g_sca = torch.Generator(device=dev).manual_seed(2_000 + mi)
            g_det = torch.Generator(device=dev).manual_seed(3_000 + mi)
            acc = None
            while acc is None or acc[0].shape[0] < maxb:
                need = maxb - (0 if acc is None else acc[0].shape[0])
                r, ni = transport(*gen_muons(int(min(need, 200_000)*2.9),
                                             args.side, dev, g_src),
                                  inv_X0, args.side, args.nvox, args.nstep, g_sca)
                n_inc_tot += ni
                if r is None:
                    continue
                acc = r if acc is None else tuple(torch.cat([a, x])
                                                  for a, x in zip(acc, r))
            n_acc_tot += acc[0].shape[0]
            for b in args.budgets:
                tr = tuple(x[:b] for x in acc)
                if args.det_sigma > 0:
                    tr = apply_detector(tr, args.det_sigma/1000.0,
                                        args.plane_near, args.plane_far, g_det)
                pt, ok = poca(tr[0], tr[1], tr[2], tr[3])
                lam, cnt = reconstruct(pt, tr[5], tr[4], ok, args.side,
                                       args.nvox, not args.no_momentum)
                sc, loc = scene_score(lam, cnt, args.side, args.nvox)
                err = float("nan")
                if truth["material"] is not None and sc == sc:
                    err = sum((a-b_)**2 for a, b_ in zip(loc, truth["center"]))**0.5
                rows.append(dict(scene=si, seed=mi, budget=b, score=sc,
                                 label=int(truth["present"]),
                                 material=truth["material"] or "none",
                                 size=truth["size"], fill=truth["fill"],
                                 loc_err=err))
        if (si+1) % 10 == 0:
            print(f"  {si+1}/{args.n_scenes} scenes  {time.time()-t0:6.1f}s")

    acceptance = n_acc_tot/n_inc_tot
    rate = (args.side*100)**2*FLUX_PER_CM2_MIN*acceptance
    npos = sum(1 for r in rows if r["label"] == 1 and r["budget"] == args.budgets[0])
    print(f"\nacceptance {acceptance:.4f} -> {rate:,.0f} accepted/min")
    print(f"scene-seed draws: {npos//args.n_seeds} threat / "
          f"{args.n_scenes - npos//args.n_seeds} benign\n")

    print(f"{'budget':>9} {'scan(min)':>10} {'sceneAUC':>9} {'sd(scene)':>10} "
          f"{'sd(muon)':>9} {'locErr(m)':>10}")
    for b in args.budgets:
        sub = [r for r in rows if r["budget"] == b]
        auc = roc_auc([r["score"] for r in sub], [r["label"] for r in sub])
        # crossed variance decomposition on the scene score
        per_scene = {}
        for r in sub:
            per_scene.setdefault(r["scene"], []).append(r["score"])
        means = torch.tensor([sum(v)/len(v) for v in per_scene.values()])
        within = torch.tensor([max(v)-min(v) for v in per_scene.values()]).mean()
        le = [r["loc_err"] for r in sub if r["loc_err"] == r["loc_err"]]
        print(f"{b:9,} {b/rate:10.1f} {auc:9.3f} {means.std():10.3f} "
              f"{within/2:9.3f} {sum(le)/max(len(le),1):10.3f}")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save, rows=np.array(
            [(r["scene"], r["seed"], r["budget"], r["score"], r["label"],
              r["size"], r["fill"], r["loc_err"]) for r in rows]),
            cols=np.array(["scene","seed","budget","score","label","size",
                           "fill","loc_err"]),
            materials=np.array([r["material"] for r in rows]))
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# Protocol, to be honored:
#   1. Kernel is 3^3, frozen. If swept (1/3/5/7), sweep on DEV ONLY.
#   2. Alarm threshold calibrated on CAL negatives for a stated FPR.
#   3. TEST touched once, with kernel and threshold already fixed.
# Violating this turns the statistic into a fitted hyperparameter and the
# reported FPR becomes meaningless.
#
# Known limits carried forward from muon_spike.py:
#   * Not validated against Geant4. This is a fast surrogate, not an oracle.
#   * Momentum spectrum is a crude p^-2.7; replace with CRY/EcoMug.
#   * Scattering only, no absorption modality.
#   * Detector: position noise only. No efficiency, no plane scattering.
#     1 mm at a 1 m gap = 1.41 mrad is OPTIMISTIC; fielded compact systems
#     report ~2.5 mm and ~8.7 mrad at 40 cm separation. Sweep sigma and gap.
#   * Targets are axis-aligned cubes. Real threats are not.
#   * clutter fill loop counts requested block volume, not unique voxels, so
#     realized fill varies (~26-29%). It is recorded per scene; stratify by it.
