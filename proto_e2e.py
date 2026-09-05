"""End-to-end muon scattering tomography prototype (numpy) -> tracks-to-detection."""
import numpy as np

rng = np.random.default_rng(1)
X0_CM = {"air": 30400.0, "steel": 1.757, "tungsten": 0.3504}

L = 1.0                  # volume side (m)
NVOX = 50                # voxels per side -> 2 cm
VOX = L / NVOX
NSTEP = 100
P_REF = 3000.0           # MeV
CUBE = 0.10              # tungsten cube side (m)

def build_phantom():
    """Voxel grid of inverse radiation length (1/m). Air + centered W cube."""
    inv_X0 = np.full((NVOX, NVOX, NVOX), 100.0 / X0_CM["air"], dtype=np.float32)
    c = np.arange(NVOX) * VOX - L/2 + VOX/2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    cube = (np.abs(X) < CUBE/2) & (np.abs(Y) < CUBE/2) & (np.abs(Z) < CUBE/2)
    inv_X0[cube] = 100.0 / X0_CM["tungsten"]
    return inv_X0, cube

def gen_muons(n):
    """Muons crossing the top face. Zenith pdf ~ cos^3(th)sin(th) (flux through
    a horizontal plane). Returns entry point on z=+L/2 and unit direction."""
    u = rng.random(n)
    th_max = np.pi/3
    cth = (1.0 - u*(1.0 - np.cos(th_max)**4)) ** 0.25
    sth = np.sqrt(1.0 - cth**2)
    phi = rng.random(n) * 2*np.pi
    d = np.stack([sth*np.cos(phi), sth*np.sin(phi), -cth], -1)
    entry = np.stack([rng.random(n)*L - L/2, rng.random(n)*L - L/2,
                      np.full(n, L/2)], -1)
    p = 1000.0 * (1.0 - rng.random(n)) ** (-1.0/1.7)   # p^-2.7 above 1 GeV
    return entry.astype(np.float64), d.astype(np.float64), p

def transport(entry, d, p, inv_X0):
    """March through the grid, accumulate MCS. Small-angle superposition:
    total exit angle = sum of step kicks; lateral offset = sum(kick * lever arm)."""
    t_exit = (entry[:, 2] + L/2) / (-d[:, 2])          # to z = -L/2
    exit_pt = entry + t_exit[:, None] * d
    inside = (np.abs(exit_pt[:, 0]) < L/2) & (np.abs(exit_pt[:, 1]) < L/2)
    entry, d, p, t_exit = entry[inside], d[inside], p[inside], t_exit[inside]
    n = len(p)
    if n == 0:
        return None

    s = (np.arange(NSTEP) + 0.5) / NSTEP                # fractional arc position
    pts = entry[:, None, :] + (t_exit[:, None] * s)[:, :, None] * d[:, None, :]
    idx = np.clip(((pts + L/2) / VOX).astype(np.int32), 0, NVOX-1)
    inv = inv_X0[idx[..., 0], idx[..., 1], idx[..., 2]]  # (n, NSTEP) 1/m

    step_len = t_exit / NSTEP                            # m
    X_step = inv * step_len[:, None]                     # rad lengths per step
    X_tot = X_step.sum(1)
    logf = 1.0 + 0.038*np.log(np.maximum(X_tot, 1e-12))
    logf = np.maximum(logf, 0.0)
    th_step = (13.6/p)[:, None] * np.sqrt(X_step) * logf[:, None]

    kx = rng.normal(size=(n, NSTEP)) * th_step
    ky = rng.normal(size=(n, NSTEP)) * th_step
    lever = t_exit[:, None] * (1.0 - s)                  # remaining path
    dx, dy = (kx*lever).sum(1), (ky*lever).sum(1)
    tx, ty = kx.sum(1), ky.sum(1)

    # build outgoing ray in the frame of the incoming direction
    zh = d
    a = np.tile(np.array([1.0, 0, 0]), (n, 1))
    a[np.abs(zh[:, 0]) > 0.9] = np.array([0, 1.0, 0])
    xh = np.cross(a, zh); xh /= np.linalg.norm(xh, axis=1, keepdims=True)
    yh = np.cross(zh, xh)
    d_out = zh + tx[:, None]*xh + ty[:, None]*yh
    d_out /= np.linalg.norm(d_out, axis=1, keepdims=True)
    p_out = entry + t_exit[:, None]*d + dx[:, None]*xh + dy[:, None]*yh
    theta = np.sqrt(tx**2 + ty**2)
    return entry, d, p_out, d_out, p, theta

def poca(p1, d1, p2, d2):
    w0 = p1 - p2
    a = (d1*d1).sum(1); b = (d1*d2).sum(1); c = (d2*d2).sum(1)
    dd = (d1*w0).sum(1); e = (d2*w0).sum(1)
    den = a*c - b*b
    ok = np.abs(den) > 1e-14
    den = np.where(ok, den, 1.0)
    sc = (b*e - c*dd)/den; tc = (a*e - b*dd)/den
    pt = 0.5*((p1 + sc[:, None]*d1) + (p2 + tc[:, None]*d2))
    return pt, ok

def reconstruct(pt, theta, p, ok):
    """Median momentum-corrected theta^2 per voxel."""
    inside = ok & (np.abs(pt) < L/2).all(1)
    pt, theta, p = pt[inside], theta[inside], p[inside]
    v = ((pt + L/2)/VOX).astype(np.int32)
    v = np.clip(v, 0, NVOX-1)
    lin = (v[:, 0]*NVOX + v[:, 1])*NVOX + v[:, 2]
    metric = (theta * (p/P_REF))**2 * 1e6                # mrad^2
    order = np.argsort(lin, kind="stable")
    lin_s, m_s = lin[order], metric[order]
    uniq, start, cnt = np.unique(lin_s, return_index=True, return_counts=True)
    lam = np.zeros(NVOX**3); count = np.zeros(NVOX**3)
    for u, st, cn in zip(uniq, start, cnt):
        lam[u] = np.median(m_s[st:st+cn]); count[u] = cn
    return lam.reshape(NVOX, NVOX, NVOX), count.reshape(NVOX, NVOX, NVOX)

def detect(lam, count, cube, min_count=2):
    valid = count >= min_count
    t = lam[cube & valid]
    c = np.arange(NVOX)*VOX - L/2 + VOX/2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    far = (np.abs(X) > CUBE) | (np.abs(Y) > CUBE) | (np.abs(Z) > CUBE)
    b = lam[far & valid]
    if len(t) < 5 or len(b) < 50:
        return np.nan, np.nan, len(t)
    mad = np.median(np.abs(b - np.median(b))) * 1.4826
    cnr = (np.median(t) - np.median(b)) / max(mad, 1e-9)
    allv = np.concatenate([t, b]); lab = np.concatenate([np.ones(len(t)), np.zeros(len(b))])
    r = np.argsort(np.argsort(allv)) + 1
    auc = (r[lab == 1].sum() - len(t)*(len(t)+1)/2) / (len(t)*len(b))
    return cnr, auc, len(t)

if __name__ == "__main__":
    inv_X0, cube = build_phantom()
    print(f"Phantom: {NVOX}^3 voxels @ {VOX*100:.0f} cm | W cube {CUBE*100:.0f} cm "
          f"({cube.sum()} voxels)\n")
    print(f"{'gen':>9} {'used':>8} {'scan(min)':>10} {'CNR':>8} {'AUC':>7}")

    acc = None
    for target in [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]:
        have = 0 if acc is None else len(acc[0])
        need = target - have
        while need > 0:
            b = min(need, 100_000)
            r = transport(*gen_muons(int(b*2.9)), inv_X0)
            if r is None: continue
            r = tuple(x[:min(len(x), need)] for x in r)
            acc = r if acc is None else tuple(np.concatenate([a, x]) for a, x in zip(acc, r))
            need = target - len(acc[0])
        e, d, po, do, p, th = acc
        pt, ok = poca(e, d, po, do)
        lam, cnt = reconstruct(pt, th, p, ok)
        cnr, auc, nt = detect(lam, cnt, cube)
        print(f"{target:9,} {len(e):8,} {len(e)/10000:10.1f} {cnr:8.2f} {auc:7.3f}")
