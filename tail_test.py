"""Target sits at 99.9th pct but clutter maxima exceed it. Hypothesis: the
field tail is driven by low-count voxels with lucky low-momentum muons.
Levers: min_count (stabilizes per-voxel median) and momentum (removes the
p^-2.7 tail). Test both at scene level."""
import numpy as np, proto_e2e as P
from scipy.ndimage import uniform_filter, label
import stat_compare as SC

BUD = 250_000
NS = 24
MC = (2, 10, 25)

def field(lam, cnt, mc, k=3, min_sup=8):
    v = (cnt >= mc).astype(np.float64)
    num = uniform_filter(lam*v, k, mode="constant")*k**3
    den = uniform_filter(v, k, mode="constant")*k**3
    return np.where(den >= min_sup, num/np.maximum(den,1.0), np.nan)

def two_stats(f):
    fin = f[np.isfinite(f)]
    if fin.size < 500: return np.nan, np.nan, 0
    med = np.median(fin); mad = np.median(np.abs(fin-med))*1.4826
    thr = np.percentile(fin, 99.9)
    lab,_ = label(np.nan_to_num(f, nan=-np.inf) > thr)
    sizes = np.bincount(lab.ravel())[1:] if lab.max()>0 else np.array([0])
    return (fin.max()-med)/max(mad,1e-9), float(sizes.max()), fin.size

keys = [(mom, mc) for mom in (False, True) for mc in MC]
B = {k: [] for k in keys}; D = {k: [] for k in keys}; NV = {k: [] for k in keys}
ys = []
for si in range(NS):
    inv, y = SC.scene(500+si); ys.append(y)
    acc=None
    while acc is None or len(acc[0])<BUD:
        r = P.transport(*P.gen_muons(200_000), inv)
        if r is None: continue
        acc = r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
    a = tuple(x[:BUD] for x in acc)
    pt, ok = P.poca(a[0],a[1],a[2],a[3])
    for mom in (False, True):
        mvec = a[4] if mom else np.full(BUD, P.P_REF)
        lam, cnt = P.reconstruct(pt, a[5], mvec, ok)
        for mc in MC:
            b, d, nv = two_stats(field(lam, cnt, mc))
            B[(mom,mc)].append(b); D[(mom,mc)].append(d); NV[(mom,mc)].append(nv)
    if (si+1)%8==0: print(f"  {si+1}/{NS}")

print(f"\nscene-level AUC at {BUD:,} tracks, {NS} scenes "
      f"({sum(ys)} threat / {NS-sum(ys)} benign)\n")
print(f"{'momentum':>9} {'min_count':>10} {'n_valid':>9} "
      f"{'B_madz':>8} {'D_clust':>9}")
for mom in (False, True):
    for mc in MC:
        k = (mom, mc)
        print(f"{'yes' if mom else 'NO':>9} {mc:10d} {int(np.mean(NV[k])):9,} "
              f"{SC.auc(B[k],ys):8.3f} {SC.auc(D[k],ys):9.3f}")
