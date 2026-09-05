"""Is the target region elevated at all, or is max() chasing the tail?"""
import numpy as np, proto_e2e as P
from scipy.ndimage import uniform_filter

rng = np.random.default_rng(11)
N_V, VOX, L = P.NVOX, P.VOX, P.L

def scene(seed, size=0.10, mat="tungsten", fill=0.35):
    r = np.random.default_rng(seed)
    inv = np.full((N_V,)*3, 100.0/P.X0_CM["air"], np.float32)
    placed = 0
    while placed < fill*N_V**3:
        sz = r.integers(4,14,3); o = r.integers(0,N_V-sz,3)
        m = "steel" if r.random() < 0.45 else "water"
        inv[o[0]:o[0]+sz[0],o[1]:o[1]+sz[1],o[2]:o[2]+sz[2]] = 100.0/{"steel":1.757,"water":36.08}[m]
        placed += int(np.prod(sz))
    marg = size/2+0.05
    ctr = r.random(3)*(L-2*marg) - (L/2-marg)
    c = np.arange(N_V)*VOX - L/2 + VOX/2
    X,Y,Z = np.meshgrid(c,c,c,indexing="ij")
    msk = ((np.abs(X-ctr[0])<size/2)&(np.abs(Y-ctr[1])<size/2)&(np.abs(Z-ctr[2])<size/2))
    inv[msk] = 100.0/P.X0_CM[mat]
    return inv, ctr, msk

def field_of(lam, cnt, min_count=2, k=3, min_sup=8):
    valid = (cnt >= min_count).astype(np.float64)
    num = uniform_filter(lam*valid, k, mode="constant")*k**3
    den = uniform_filter(valid, k, mode="constant")*k**3
    f = np.where(den >= min_sup, num/np.maximum(den,1.0), -np.inf)
    return f

for budget in (50_000, 250_000):
    print(f"\n===== budget {budget:,} =====")
    for trial in range(3):
        inv, ctr, msk = scene(200+trial)
        acc=None
        while acc is None or len(acc[0])<budget:
            r = P.transport(*P.gen_muons(200_000), inv)
            if r is None: continue
            acc = r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
        a = tuple(x[:budget] for x in acc)
        pt, ok = P.poca(a[0],a[1],a[2],a[3])
        lam, cnt = P.reconstruct(pt, a[5], np.full(budget,P.P_REF), ok)
        f = field_of(lam, cnt)
        fin = f[np.isfinite(f)]
        # value of the field at the true target
        f_tgt = f[msk & np.isfinite(f)]
        gm = np.unravel_index(np.argmax(f), f.shape)
        c = np.arange(N_V)*VOX - L/2 + VOX/2
        gloc = np.array([c[gm[0]],c[gm[1]],c[gm[2]]])
        pct = 100.0*(fin < (f_tgt.max() if f_tgt.size else -np.inf)).mean()
        print(f" trial {trial}: n_valid={fin.size:6d}  "
              f"tgt_vox_valid={f_tgt.size:3d}/{msk.sum()}")
        print(f"   field: median={np.median(fin):8.1f} p99={np.percentile(fin,99):9.1f} "
              f"max={fin.max():10.1f}")
        print(f"   at target: max={f_tgt.max() if f_tgt.size else float('nan'):9.1f} "
              f"-> percentile {pct:5.2f}   argmax err={np.linalg.norm(gloc-ctr):.3f} m")
        # how many voxels in target actually got tracks
        print(f"   tracks in target voxels: {int(cnt[msk].sum())}, "
              f"median cnt/voxel={np.median(cnt[msk]):.0f}")
