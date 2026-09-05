#!/usr/bin/env python3
"""
muon_spike.py v2 -- cosmic-ray muon scattering tomography, vectorized on GPU.

Changes from v1:
  * FIXED: scan time divided accepted tracks by the INCIDENT rate, ignoring
    geometric acceptance. Times were optimistic by 1/acceptance (~2.83x here).
    Acceptance is now measured at runtime and reported separately.
  * --no-momentum   : drop per-track momentum (real portals mostly lack it)
  * --det-sigma     : detector position resolution (mm), with direction refit
                      from two tracking planes per side
  * reports n_valid : voxels meeting min_count, so support drift is visible

    python muon_spike.py
    python muon_spike.py --no-momentum --det-sigma 1.0
    python muon_spike.py --phantom simple --max-tracks 4000000 --save recon.npz

Read NOTES at the bottom before quoting any number from this file.
"""
import argparse, time
import torch

X0_CM = {"air": 30400.0, "steel": 1.757, "tungsten": 0.3504, "water": 36.08,
         "lead": 0.5612, "uranium": 0.3166}
P_REF_MEV = 3000.0
FLUX_PER_CM2_MIN = 1.0      # sea-level rate through a horizontal plane


def build_phantom(kind, nvox, side, cube_m, dev, seed=3, fill=0.35):
    g = torch.Generator(device="cpu").manual_seed(seed)
    inv = torch.full((nvox,) * 3, 100.0 / X0_CM["air"])
    if kind == "cluttered":
        placed = 0
        while placed < fill * nvox ** 3:
            sz = torch.randint(4, 14, (3,), generator=g)
            o = torch.stack([torch.randint(0, nvox - int(s), (1,), generator=g)[0]
                             for s in sz])
            mat = "steel" if torch.rand(1, generator=g).item() < 0.45 else "water"
            inv[o[0]:o[0]+sz[0], o[1]:o[1]+sz[1], o[2]:o[2]+sz[2]] = 100.0/X0_CM[mat]
            placed += int(sz.prod())
    vox = side / nvox
    c = torch.arange(nvox) * vox - side / 2 + vox / 2
    X, Y, Z = torch.meshgrid(c, c, c, indexing="ij")
    target = ((X.abs() < cube_m/2) & (Y.abs() < cube_m/2) & (Z.abs() < cube_m/2))
    inv[target] = 100.0 / X0_CM["tungsten"]
    return inv.to(dev), target.to(dev)


def gen_muons(n, side, dev, th_max=torch.pi / 3):
    """Zenith pdf ~ cos^3(th)sin(th): cos^2 intensity times cos for the flux
    through a horizontal plane."""
    u = torch.rand(n, device=dev, dtype=torch.float64)
    cmax = torch.cos(torch.tensor(th_max, dtype=torch.float64)) ** 4
    cth = (1.0 - u * (1.0 - cmax)) ** 0.25
    sth = torch.sqrt(1.0 - cth ** 2)
    phi = torch.rand(n, device=dev, dtype=torch.float64) * 2 * torch.pi
    d = torch.stack([sth*torch.cos(phi), sth*torch.sin(phi), -cth], -1)
    entry = torch.stack([
        torch.rand(n, device=dev, dtype=torch.float64) * side - side/2,
        torch.rand(n, device=dev, dtype=torch.float64) * side - side/2,
        torch.full((n,), side/2, device=dev, dtype=torch.float64)], -1)
    p = 1000.0 * (1.0 - torch.rand(n, device=dev, dtype=torch.float64)) ** (-1/1.7)
    return entry, d, p


def transport(entry, d, p, inv_X0, side, nvox, nstep):
    """Ray-march + Highland MCS, small-angle superposition of per-step kicks.
    Returns (tracks, n_incident) so acceptance can be accounted correctly."""
    n_inc = entry.shape[0]
    vox = side / nvox
    t_exit = (entry[:, 2] + side/2) / (-d[:, 2])
    exit_pt = entry + t_exit[:, None] * d
    keep = (exit_pt[:, 0].abs() < side/2) & (exit_pt[:, 1].abs() < side/2)
    if not keep.any():
        return None, n_inc
    entry, d, p, t_exit = entry[keep], d[keep], p[keep], t_exit[keep]
    n, dev = p.shape[0], p.device

    s = (torch.arange(nstep, device=dev, dtype=torch.float64) + 0.5) / nstep
    pts = entry[:, None, :] + (t_exit[:, None] * s)[:, :, None] * d[:, None, :]
    idx = (((pts + side/2) / vox).long()).clamp_(0, nvox - 1)
    inv = inv_X0[idx[..., 0], idx[..., 1], idx[..., 2]].double()
    del pts, idx

    X_step = inv * (t_exit / nstep)[:, None]
    X_tot = X_step.sum(1)
    # PDG log correction uses the TOTAL path, so per-step variances add exactly
    # to the single-shot Highland result.
    logf = (1.0 + 0.038 * torch.log(X_tot.clamp_min(1e-12))).clamp_min(0.0)
    th_step = (13.6 / p)[:, None] * torch.sqrt(X_step) * logf[:, None]

    kx = torch.randn(n, nstep, device=dev, dtype=torch.float64) * th_step
    ky = torch.randn(n, nstep, device=dev, dtype=torch.float64) * th_step
    lever = t_exit[:, None] * (1.0 - s)
    dx, dy = (kx*lever).sum(1), (ky*lever).sum(1)
    tx, ty = kx.sum(1), ky.sum(1)
    del kx, ky, th_step, X_step, inv, lever

    a = torch.zeros_like(d); a[:, 0] = 1.0
    a[d[:, 0].abs() > 0.9] = torch.tensor([0., 1., 0.], device=dev, dtype=torch.float64)
    xh = torch.cross(a, d, dim=1); xh = xh / xh.norm(dim=1, keepdim=True)
    yh = torch.cross(d, xh, dim=1)
    d_out = d + tx[:, None]*xh + ty[:, None]*yh
    d_out = d_out / d_out.norm(dim=1, keepdim=True)
    p_out = entry + t_exit[:, None]*d + dx[:, None]*xh + dy[:, None]*yh
    return (entry, d, p_out, d_out, p, torch.sqrt(tx**2 + ty**2)), n_inc


def apply_detector(tr, sigma_m, z_near, z_far):
    """Refit in/out directions from two noisy tracking planes per side.
    Returns measured (pt_in, dir_in, pt_out, dir_out, momentum, theta)."""
    entry, d, p_out, d_out, p, _ = tr

    def refit(pt, dv, za, zb):
        ta = (za - pt[:, 2]) / dv[:, 2]
        tb = (zb - pt[:, 2]) / dv[:, 2]
        A = pt + ta[:, None]*dv
        B = pt + tb[:, None]*dv
        A[:, :2] += torch.randn_like(A[:, :2]) * sigma_m
        B[:, :2] += torch.randn_like(B[:, :2]) * sigma_m
        nd = B - A
        nd = nd * torch.sign((nd*dv).sum(1))[:, None]   # preserve travel sense
        return A, nd / nd.norm(dim=1, keepdim=True)

    Ain, din = refit(entry, d, z_near, z_far)
    Aout, dout = refit(p_out, d_out, -z_near, -z_far)
    theta = torch.arccos((din*dout).sum(1).clamp(-1, 1))
    return Ain, din, Aout, dout, p, theta


def poca(p1, d1, p2, d2):
    w0 = p1 - p2
    a = (d1*d1).sum(1); b = (d1*d2).sum(1); c = (d2*d2).sum(1)
    dd = (d1*w0).sum(1); e = (d2*w0).sum(1)
    den = a*c - b*b
    ok = den.abs() > 1e-14
    den = torch.where(ok, den, torch.ones_like(den))
    sc = (b*e - c*dd)/den; tc = (a*e - b*dd)/den
    return 0.5*((p1 + sc[:, None]*d1) + (p2 + tc[:, None]*d2)), ok


def reconstruct(pt, theta, p, ok, side, nvox, use_momentum=True):
    vox = side / nvox
    inside = ok & (pt.abs() < side/2).all(1)
    pt, theta, p = pt[inside], theta[inside], p[inside]
    v = (((pt + side/2)/vox).long()).clamp_(0, nvox-1)
    lin = (v[:, 0]*nvox + v[:, 1])*nvox + v[:, 2]
    scale = (p / P_REF_MEV) if use_momentum else torch.ones_like(p)
    metric = (theta * scale)**2 * 1e6                    # mrad^2

    o1 = torch.argsort(metric)
    lin_s, m_s = lin[o1], metric[o1]
    o2 = torch.argsort(lin_s, stable=True)
    lin_s, m_s = lin_s[o2], m_s[o2]
    uniq, counts = torch.unique_consecutive(lin_s, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=counts.device),
                        counts.cumsum(0)[:-1]])
    lam = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    cnt = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    lam[uniq] = m_s[starts + counts // 2]
    cnt[uniq] = counts.double()
    return lam.reshape((nvox,)*3), cnt.reshape((nvox,)*3)


def detect(lam, cnt, target, side, nvox, cube_m, min_count=2):
    vox = side / nvox
    c = torch.arange(nvox, device=lam.device)*vox - side/2 + vox/2
    X, Y, Z = torch.meshgrid(c, c, c, indexing="ij")
    far = (X.abs() > cube_m) | (Y.abs() > cube_m) | (Z.abs() > cube_m)
    valid = cnt >= min_count
    t, b = lam[target & valid], lam[far & valid]
    n_valid = int(valid.sum())
    if t.numel() < 5 or b.numel() < 50:
        return float("nan"), float("nan"), n_valid
    mad = (b - b.median()).abs().median() * 1.4826
    cnr = ((t.median() - b.median()) / mad.clamp_min(1e-9)).item()
    allv = torch.cat([t, b])
    lab = torch.cat([torch.ones_like(t), torch.zeros_like(b)])
    r = torch.argsort(torch.argsort(allv)).double() + 1
    nt, nb = t.numel(), b.numel()
    auc = ((r[lab == 1].sum() - nt*(nt+1)/2) / (nt*nb)).item()
    return cnr, auc, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=["cluttered", "simple"], default="cluttered")
    ap.add_argument("--nvox", type=int, default=50)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--cube", type=float, default=0.10)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--max-tracks", type=int, default=2_000_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--no-momentum", action="store_true",
                    help="portal cannot measure per-track momentum")
    ap.add_argument("--det-sigma", type=float, default=0.0,
                    help="detector position resolution in mm (0 = perfect)")
    ap.add_argument("--plane-near", type=float, default=0.6)
    ap.add_argument("--plane-far", type=float, default=1.6)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        cc = torch.cuda.get_device_capability()
        print(f"device: {torch.cuda.get_device_name()}  sm_{cc[0]}{cc[1]}  "
              f"torch {torch.__version__}")
    else:
        print(f"device: cpu  torch {torch.__version__}")

    inv_X0, target = build_phantom(args.phantom, args.nvox, args.side,
                                   args.cube, dev, seed=3)
    nonair = (inv_X0 > 1.0).double().mean().item()*100
    incident_rate = (args.side*100)**2 * FLUX_PER_CM2_MIN
    print(f"phantom: {args.phantom}, {args.nvox}^3 @ {args.side/args.nvox*100:.1f} cm, "
          f"{nonair:.0f}% non-air, W target {args.cube*100:.0f} cm "
          f"({int(target.sum())} vox)")
    print(f"model: momentum={'NO' if args.no_momentum else 'yes'}, "
          f"det sigma={args.det_sigma} mm", end="")
    if args.det_sigma > 0:
        gap = args.plane_far - args.plane_near
        print(f" (planes {gap:.1f} m apart -> "
              f"sig_theta={args.det_sigma*1e-3*2**0.5/gap*1e3:.2f} mrad)")
    else:
        print()

    sweep = [n for n in (10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
                         1_000_000, 2_000_000, 4_000_000) if n <= args.max_tracks]
    acc, n_incident, t0 = None, 0, time.time()
    rows = []
    for tgt in sweep:
        while acc is None or acc[0].shape[0] < tgt:
            need = tgt - (0 if acc is None else acc[0].shape[0])
            r, ni = transport(*gen_muons(int(min(need, args.chunk)*2.9), args.side, dev),
                              inv_X0, args.side, args.nvox, args.nstep)
            n_incident += ni
            if r is None:
                continue
            acc = r if acc is None else tuple(torch.cat([a, x]) for a, x in zip(acc, r))
        tr = tuple(x[:tgt] for x in acc)
        if args.det_sigma > 0:
            tr = apply_detector(tr, args.det_sigma/1000.0,
                                args.plane_near, args.plane_far)
        pt, ok = poca(tr[0], tr[1], tr[2], tr[3])
        lam, cnt = reconstruct(pt, tr[5], tr[4], ok, args.side, args.nvox,
                               use_momentum=not args.no_momentum)
        cnr, auc, nv = detect(lam, cnt, target, args.side, args.nvox,
                              args.cube, args.min_count)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        rows.append((tgt, cnr, auc, nv, time.time() - t0))

    acceptance = acc[0].shape[0] / n_incident
    accepted_rate = incident_rate * acceptance
    print(f"\ngeometric acceptance: {acceptance:.4f}  -> accepted rate "
          f"{accepted_rate:,.0f}/min (incident {incident_rate:,.0f}/min)\n")
    print(f"{'tracks':>10} {'scan(min)':>10} {'CNR':>8} {'AUC':>7} "
          f"{'n_valid':>9} {'s':>7}")
    for tgt, cnr, auc, nv, el in rows:
        print(f"{tgt:10,} {tgt/accepted_rate:10.1f} {cnr:8.2f} {auc:7.3f} "
              f"{nv:9,} {el:7.1f}")

    if args.save:
        import numpy as np
        np.savez_compressed(args.save, lam=lam.cpu().numpy(),
                            count=cnt.cpu().numpy(), target=target.cpu().numpy(),
                            acceptance=acceptance)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# Remaining idealizations, roughly in order of how much they could hurt:
#
# 1. Voxel AUC in a single target-present scene is NOT the operational
#    detection problem. Target position is fixed and known to the scorer.
#    Real question: unknown location, unknown material, sometimes no target
#    at all -> object-level ROC across many scenes. Until that exists, AUC
#    here is an upper bound and should never be quoted as detection
#    performance.
# 2. Support drift: n_valid changes with track count, so AUC at 10k and at
#    1M are computed over different voxel sets. Any stopping rule reading
#    this statistic is reading a moving target. Fix before building policy.
# 3. Detector planes are ideal apart from position noise: no efficiency
#    loss, no scattering in the planes themselves, no dead area.
# 4. Momentum spectrum is a crude p^-2.7 with a 1 GeV floor. Replace with
#    CRY or EcoMug before any number leaves the building.
# 5. Scattering only. No absorption/stopping, which the literature combines
#    with scattering as a second modality.
# 6. Zenith cut at 60 deg; real flux extends toward horizontal.
# 7. Small-angle superposition ignores within-step lateral offset.
# 8. The 1 m cube geometry has poor acceptance (~0.35) because it is as tall
#    as it is wide. A real container is much wider than tall and will accept
#    a larger fraction. Do not port this acceptance number to other geometry.
# 9. Flux normalization 1 muon/cm^2/min. Some cargo papers quote ~1M
#    muons/min for a container, ~7x this. Resolve before citing scan times.
#
# Validated: Highland closed form (10 cm W -> 27.3 mrad at 3 GeV); per-step
# variance additivity exact to 1e-6; PoCA recovers an analytic kink to
# 1e-13 m; acceptance 0.3536 measured over 1.2M incident muons. NOT validated
# against Geant4 -- that remains the next gate.
