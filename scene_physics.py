#!/usr/bin/env python3
"""
scene_physics.py -- Phase 1. Separate the world from our representation of it.

INVARIANT THIS FILE EXISTS TO ESTABLISH:
    Changing reconstruction resolution must not change the physical scene,
    the generated muons, or the scattering history.

Layers, strictly ordered, each ignorant of the one below:
    manifest   physical boxes in METRES. No nvox anywhere.
    forward    analytic point-in-box material lookup. No voxel grid.
    events     immutable per-muon record. Continuous coordinates.
    recon      the ONLY layer that knows what nvox means.

Geant4 later becomes another producer of the same event record.

    python scene_physics.py              # run the invariance suite
    python scene_physics.py --n-scenes 400 --save events_dev.npz

No AUC here by design. Scoring is Phase 2.
"""
import argparse, hashlib, time
import torch

X0_M = {"air": 304.0, "steel": 0.01757, "water": 0.3608,
        "lead": 0.005612, "tungsten": 0.003504, "uranium": 0.003166}
THREAT_MATS = ["lead", "tungsten", "uranium"]
TARGET_SIZES_M = [0.06, 0.08, 0.10, 0.14]
CLUTTER_MATS = ["steel", "water"]
N_CLUTTER_ATTEMPTS = 1200         # fixed: RNG consumption must not vary
BLOCK_MIN_M, BLOCK_MAX_M = 0.08, 0.28


# ---------------------------------------------------------------- manifest
def make_manifest(scene_seed, side=1.0):
    """Physical scene in metres. Depends ONLY on scene_seed.

    Independent sub-streams so clutter draws can never shift the target draw
    -- that coupling is what broke the previous resolution comparison.
    """
    gc = torch.Generator().manual_seed(scene_seed * 2 + 1)
    gt = torch.Generator().manual_seed(scene_seed * 2 + 2)

    boxes = []                                    # mutually non-overlapping
    C = torch.zeros(0, 3); H = torch.zeros(0, 3)   # stacked, for vector reject
    for _ in range(N_CLUTTER_ATTEMPTS):
        h = (torch.rand(3, generator=gc) * (BLOCK_MAX_M - BLOCK_MIN_M)
             + BLOCK_MIN_M) / 2
        c = (torch.rand(3, generator=gc) * (side - 2*h)) - (side/2 - h)
        mi = int(torch.rand(1, generator=gc).item() * len(CLUTTER_MATS))
        # AABBs overlap iff they overlap on every axis
        clash = (len(boxes) > 0 and
                 bool((((c - C).abs() < (h + H)).all(1)).any()))
        if not clash:
            boxes.append(dict(c=c, h=h, mat=CLUTTER_MATS[mi], kind="clutter"))
            C = torch.cat([C, c[None]]); H = torch.cat([H, h[None]])

    u = torch.rand(4, generator=gt)
    present = bool(u[0] < 0.5)
    if present:
        mat = THREAT_MATS[int(u[1] * len(THREAT_MATS))]
    elif u[1] < 0.5:
        mat = "steel"                             # benign dense confuser
    else:
        mat = None
    size = TARGET_SIZES_M[int(u[2] * len(TARGET_SIZES_M))]

    target = None
    if mat is not None:
        h = torch.full((3,), size/2)
        lim = side/2 - size/2 - 0.02
        c = torch.rand(3, generator=gt) * (2*lim) - lim
        target = dict(c=c, h=h, mat=mat, kind="target")

    fill = sum(float((2*b["h"]).prod()) for b in boxes) / side**3
    return dict(scene_seed=scene_seed, side=side, boxes=boxes, target=target,
                present=present, target_mat=mat, target_size=size,
                fill=fill, n_clutter=len(boxes))


def manifest_hash(m):
    hsh = hashlib.sha256()
    hsh.update(f"{m['scene_seed']}|{m['side']}|{m['present']}|"
               f"{m['target_mat']}|{m['target_size']:.6f}".encode())
    for b in m["boxes"] + ([m["target"]] if m["target"] else []):
        hsh.update(b["mat"].encode())
        hsh.update(torch.cat([b["c"], b["h"]]).double().numpy().tobytes())
    return hsh.hexdigest()[:16]


# ---------------------------------------------------------------- forward
def gen_muons(n, side, dev, g, th_max=torch.pi/3):
    """Zenith pdf ~ cos^3(th)sin(th). Independent of nvox."""
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


def inv_X0_at(pts, manifest, dev):
    """Analytic material lookup by point-in-box. No voxel grid.
    Boxes applied in manifest order, target last, so nesting works."""
    inv = torch.full(pts.shape[:-1], 1.0/X0_M["air"], device=dev,
                     dtype=torch.float64)
    seq = manifest["boxes"] + ([manifest["target"]] if manifest["target"] else [])
    for b in seq:
        c = b["c"].to(dev).double()
        h = b["h"].to(dev).double()
        inside = ((pts - c).abs() < h).all(-1)
        inv = torch.where(inside, 1.0/X0_M[b["mat"]], inv)
    return inv


def forward(entry, d, p, manifest, nstep, g_scatter):
    """Highland MCS, small-angle superposition. nstep is a PHYSICS parameter
    (path discretisation), deliberately unrelated to reconstruction nvox."""
    side = manifest["side"]
    n_inc = entry.shape[0]
    t_exit = (entry[:, 2] + side/2)/(-d[:, 2])
    ep = entry + t_exit[:, None]*d
    keep = (ep[:, 0].abs() < side/2) & (ep[:, 1].abs() < side/2)
    if not keep.any():
        return None, n_inc
    entry, d, p, t_exit = entry[keep], d[keep], p[keep], t_exit[keep]
    n, dev = p.shape[0], p.device

    s = (torch.arange(nstep, device=dev, dtype=torch.float64) + 0.5)/nstep
    pts = entry[:, None, :] + (t_exit[:, None]*s)[:, :, None]*d[:, None, :]
    inv = inv_X0_at(pts, manifest, dev)
    del pts

    X_step = inv*(t_exit/nstep)[:, None]
    logf = (1.0 + 0.038*torch.log(X_step.sum(1).clamp_min(1e-12))).clamp_min(0.0)
    th_step = (13.6/p)[:, None]*torch.sqrt(X_step)*logf[:, None]
    kx = torch.randn(n, nstep, device=dev, dtype=torch.float64,
                     generator=g_scatter)*th_step
    ky = torch.randn(n, nstep, device=dev, dtype=torch.float64,
                     generator=g_scatter)*th_step
    lever = t_exit[:, None]*(1.0-s)
    dx, dy = (kx*lever).sum(1), (ky*lever).sum(1)
    tx, ty = kx.sum(1), ky.sum(1)
    del kx, ky, th_step, X_step, inv, lever

    a = torch.zeros_like(d); a[:, 0] = 1.0
    a[d[:, 0].abs() > 0.9] = torch.tensor([0., 1., 0.], device=dev,
                                          dtype=torch.float64)
    xh = torch.cross(a, d, dim=1); xh = xh/xh.norm(dim=1, keepdim=True)
    yh = torch.cross(d, xh, dim=1)
    d_out = d + tx[:, None]*xh + ty[:, None]*yh
    d_out = d_out/d_out.norm(dim=1, keepdim=True)
    p_out = entry + t_exit[:, None]*d + dx[:, None]*xh + dy[:, None]*yh
    ev = dict(entry=entry, d_in=d, exit=p_out, d_out=d_out, p=p,
              theta=torch.sqrt(tx**2+ty**2))
    return ev, n_inc


def events_hash(ev):
    hsh = hashlib.sha256()
    for k in ("entry", "d_in", "exit", "d_out", "p", "theta"):
        hsh.update(ev[k].cpu().double().numpy().tobytes())
    return hsh.hexdigest()[:16]


def simulate(manifest, n_tracks, nstep, dev, source_seed, scatter_seed,
             chunk=100_000):
    """Produce the immutable event record. Knows nothing about nvox."""
    g_src = torch.Generator(device=dev).manual_seed(source_seed)
    g_sca = torch.Generator(device=dev).manual_seed(scatter_seed)
    acc, n_inc = None, 0
    while acc is None or acc["p"].shape[0] < n_tracks:
        need = n_tracks - (0 if acc is None else acc["p"].shape[0])
        ev, ni = forward(*gen_muons(int(min(need, chunk)*2.9), manifest["side"],
                                    dev, g_src), manifest, nstep, g_sca)
        n_inc += ni
        if ev is None:
            continue
        acc = ev if acc is None else {k: torch.cat([acc[k], ev[k]]) for k in ev}
    acc = {k: v[:n_tracks] for k, v in acc.items()}
    return acc, n_inc


# ---------------------------------------------------------------- recon
def reconstruct(ev, nvox, side, use_momentum=False, n=None):
    """The ONLY function that knows what nvox means."""
    P_REF = 3000.0
    e, di, x, do = ev["entry"], ev["d_in"], ev["exit"], ev["d_out"]
    p, th = ev["p"], ev["theta"]
    if n is not None:
        e, di, x, do, p, th = (t[:n] for t in (e, di, x, do, p, th))
    w0 = e - x
    a = (di*di).sum(1); b = (di*do).sum(1); c = (do*do).sum(1)
    dd = (di*w0).sum(1); f = (do*w0).sum(1)
    den = a*c - b*b
    ok = den.abs() > 1e-14
    den = torch.where(ok, den, torch.ones_like(den))
    sc = (b*f - c*dd)/den; tc = (a*f - b*dd)/den
    pt = 0.5*((e + sc[:, None]*di) + (x + tc[:, None]*do))

    vox = side/nvox
    inside = ok & (pt.abs() < side/2).all(1)
    pt, th, p = pt[inside], th[inside], p[inside]
    v = (((pt + side/2)/vox).long()).clamp_(0, nvox-1)
    lin = (v[:, 0]*nvox + v[:, 1])*nvox + v[:, 2]
    scale = (p/P_REF) if use_momentum else torch.ones_like(p)
    metric = (th*scale)**2*1e6
    o1 = torch.argsort(metric)
    lin_s, m_s = lin[o1], metric[o1]
    o2 = torch.argsort(lin_s, stable=True)
    lin_s, m_s = lin_s[o2], m_s[o2]
    uniq, cnts = torch.unique_consecutive(lin_s, return_counts=True)
    st = torch.cat([torch.zeros(1, dtype=cnts.dtype, device=cnts.device),
                    cnts.cumsum(0)[:-1]])
    lam = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    cnt = torch.zeros(nvox**3, device=pt.device, dtype=torch.float64)
    lam[uniq] = m_s[st + cnts//2]
    cnt[uniq] = cnts.double()
    return lam.reshape((nvox,)*3), cnt.reshape((nvox,)*3)


def rasterize(manifest, nvox, dev):
    """Diagnostic only. NEVER used by the forward model."""
    side = manifest["side"]; vox = side/nvox
    c = (torch.arange(nvox, device=dev, dtype=torch.float64)*vox
         - side/2 + vox/2)
    X, Y, Z = torch.meshgrid(c, c, c, indexing="ij")
    return inv_X0_at(torch.stack([X, Y, Z], -1), manifest, dev)


# ---------------------------------------------------------------- tests
def invariance_suite(n_scenes, n_tracks, nstep, dev, resolutions=(50, 25)):
    print(f"invariance suite: {n_scenes} scenes, {n_tracks:,} tracks, "
          f"nstep={nstep}, resolutions={resolutions}\n")
    fails, t0 = [], time.time()
    counts = {"threat": 0, "benign": 0}
    ras_err = {}
    for si in range(n_scenes):
        mans = {r: make_manifest(si) for r in resolutions}
        hs = {r: manifest_hash(m) for r, m in mans.items()}
        if len(set(hs.values())) != 1:
            fails.append(f"scene {si}: manifest differs across resolutions")
        m = mans[resolutions[0]]
        counts["threat" if m["present"] else "benign"] += 1

        evs, incs = {}, {}
        for r in resolutions:
            ev, ni = simulate(m, n_tracks, nstep, dev, 1000, 2000)
            evs[r] = events_hash(ev); incs[r] = ni
            if r == resolutions[0]:
                ev0 = ev
        if len(set(evs.values())) != 1:
            fails.append(f"scene {si}: event record differs across resolutions")
        if len(set(incs.values())) != 1:
            fails.append(f"scene {si}: incident count differs across resolutions")

        recs = {r: reconstruct(ev0, r, m["side"]) for r in resolutions}
        for r in resolutions:
            if recs[r][1].sum() <= 0:
                fails.append(f"scene {si}: empty reconstruction at nvox={r}")
        # The forward model never rasterises, so test the ANALYTIC lookup
        # directly: every point inside the target box must return the target
        # material, i.e. the target correctly overrides nested clutter.
        if m["target"] is not None:
            c = m["target"]["c"].to(dev).double()
            h = m["target"]["h"].to(dev).double()
            q = torch.rand(2000, 3, device=dev, dtype=torch.float64)
            pin = c + (q*2 - 1)*h*0.98                  # strictly inside
            got = inv_X0_at(pin, m, dev)
            want = 1.0/X0_M[m["target_mat"]]
            if not torch.allclose(got, torch.full_like(got, want)):
                bad = int((got != want).sum())
                fails.append(f"scene {si}: analytic lookup wrong for "
                             f"{bad}/2000 points inside target")
            ras_err[m["target_size"]] = ras_err.get(m["target_size"], [])
            for r in resolutions:
                grid = rasterize(m, r, dev)
                inside = ((torch.stack(torch.meshgrid(
                    *[torch.arange(r, device=dev, dtype=torch.float64)
                      * (m["side"]/r) - m["side"]/2 + m["side"]/(2*r)]*3,
                    indexing="ij"), -1) - c).abs() < h).all(-1)
                vgrid = float(inside.double().sum()*(m["side"]/r)**3)
                vtrue = float((2*h).prod())
                ras_err[m["target_size"]].append((r, vgrid/max(vtrue, 1e-12)))
        if (si+1) % 20 == 0:
            print(f"  {si+1}/{n_scenes}  {time.time()-t0:6.1f}s")

    print(f"\nscenes: {counts['threat']} threat / {counts['benign']} benign")

    if ras_err:
        print("\nrasterisation fidelity (diagnostic only -- the forward model\n"
              "is analytic and never uses a grid; this only tells you how well\n"
              "each reconstruction resolution can REPRESENT a target):")
        print(f"  {'target':>8} " + "".join(f"{'nvox='+str(r):>14}"
              for r in resolutions))
        for size in sorted(ras_err):
            row = f"  {size*100:6.0f}cm "
            for r in resolutions:
                v = [x for rr, x in ras_err[size] if rr == r]
                row += (f"{sum(v)/len(v):8.2f}x n={len(v):<3d}" if v
                        else f"{'-':>14}")
            print(row)
        print("  (1.00x = rasterised volume matches true volume)")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} violations")
    for f in fails[:15]:
        print("  " + f)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=20)
    ap.add_argument("--n-tracks", type=int, default=50_000)
    ap.add_argument("--nstep", type=int, default=100)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default=None)
    ap.add_argument("--save-tracks", type=int, default=250_000)
    args = ap.parse_args()
    dev = torch.device(args.device)
    print(f"device: {dev}  torch {torch.__version__}\n")

    ok = invariance_suite(args.n_scenes, args.n_tracks, args.nstep, dev)

    if args.save and ok:
        import numpy as np
        recs = []
        for si in range(args.n_scenes):
            m = make_manifest(si, args.side)
            ev, ni = simulate(m, args.save_tracks, args.nstep, dev, 1000, 2000)
            recs.append(dict(scene=si, mhash=manifest_hash(m),
                             ehash=events_hash(ev), present=int(m["present"]),
                             mat=m["target_mat"] or "none", size=m["target_size"],
                             fill=m["fill"], n_clutter=m["n_clutter"],
                             n_incident=ni))
        np.savez_compressed(args.save, meta=np.array(
            [(r["scene"], r["present"], r["size"], r["fill"], r["n_clutter"],
              r["n_incident"]) for r in recs]),
            mhash=np.array([r["mhash"] for r in recs]),
            ehash=np.array([r["ehash"] for r in recs]),
            mat=np.array([r["mat"] for r in recs]))
        print(f"\nsaved manifest+event hashes -> {args.save}")
    elif args.save:
        print("\nNOT saving: invariance suite failed.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------- NOTES
# What Phase 1 fixes:
#   * Clutter blocks are sized/placed in METRES, so scene semantics no longer
#     depend on nvox. Previously blocks were 4-13 VOXELS, meaning 8-26 cm at
#     2 cm and 16-52 cm at 4 cm: different physical cargo, not a coarser view.
#   * Clutter and target draw from independent sub-streams, so a change in the
#     number of clutter attempts cannot shift the threat/benign assignment.
#     That coupling produced 197/203 vs 206/194 and voided the last grid.
#   * Forward model uses analytic point-in-box lookup. No voxel grid anywhere
#     upstream of reconstruct().
#   * Clutter boxes are mutually non-overlapping by rejection; the target is
#     applied LAST so it may nest inside clutter (needed for the shielding
#     ladder: e.g. small W inside a large steel block).
#
# What Phase 1 does NOT fix, carried forward:
#   * nstep=100 still discretises the path (~1 cm steps). It is a physics
#     parameter now, independent of nvox, but it is still an approximation.
#     Sweep it once to confirm 100 is enough.
#   * Rejection sampling means realized fill varies with scene_seed; it is
#     recorded per scene. Stratify by it, do not assume it is constant.
#   * No Geant4 cross-check. This remains a fast surrogate, not an oracle.
#   * Detector effects are not applied here. They belong between events and
#     reconstruction, and are Phase 2.
#
# Phase 2 (do not start until this suite passes): occupancy-only field,
# lambda-only field, fusion, explicit no-evidence policy, n_scored/n_unscored,
# fresh cal_v2. test_v1 seeds stay sealed and are generated only after the
# pipeline freezes.
