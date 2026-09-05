"""The target IS elevated (99.9th pct). max() fails because its null grows
with n_valid, which grows with exposure. Test scene-normalized alternatives."""
import numpy as np, proto_e2e as P
from scipy.ndimage import uniform_filter, label

N_V, VOX, L = P.NVOX, P.VOX, P.L
MATS = {"steel":1.757,"water":36.08,"lead":0.5612,"tungsten":0.3504,"uranium":0.3166}

def scene(seed, fill=0.35):
    r = np.random.default_rng(seed)
    inv = np.full((N_V,)*3, 100.0/P.X0_CM["air"], np.float32)
    placed = 0
    while placed < fill*N_V**3:
        sz = r.integers(4,14,3); o = r.integers(0,N_V-sz,3)
        inv[o[0]:o[0]+sz[0],o[1]:o[1]+sz[1],o[2]:o[2]+sz[2]] = \
            100.0/MATS["steel" if r.random()<0.45 else "water"]
        placed += int(np.prod(sz))
    present = r.random() < 0.5
    mat = r.choice(["lead","tungsten","uranium"]) if present else \
          ("steel" if r.random()<0.5 else None)
    size = r.choice([0.06,0.08,0.10,0.14])
    if mat is not None:
        m0 = size/2+0.05
        ctr = r.random(3)*(L-2*m0) - (L/2-m0)
        c = np.arange(N_V)*VOX - L/2 + VOX/2
        X,Y,Z = np.meshgrid(c,c,c,indexing="ij")
        inv[(np.abs(X-ctr[0])<size/2)&(np.abs(Y-ctr[1])<size/2)&
            (np.abs(Z-ctr[2])<size/2)] = 100.0/MATS[mat]
    return inv, int(present)

def get_field(lam, cnt, min_count=2, k=3, min_sup=8):
    v = (cnt >= min_count).astype(np.float64)
    num = uniform_filter(lam*v, k, mode="constant")*k**3
    den = uniform_filter(v, k, mode="constant")*k**3
    return np.where(den >= min_sup, num/np.maximum(den,1.0), np.nan)

def stats(f):
    fin = f[np.isfinite(f)]
    if fin.size < 500: return {k: np.nan for k in
        ("A_max","B_madz","C_tail","D_clust","E_topk")}
    med, q99 = np.median(fin), np.percentile(fin,99)
    mad = np.median(np.abs(fin-med))*1.4826
    mx = fin.max()
    thr = np.percentile(fin, 99.9)
    lab,_ = label(np.nan_to_num(f, nan=-np.inf) > thr)
    sizes = np.bincount(lab.ravel())[1:] if lab.max()>0 else np.array([0])
    return dict(
        A_max   = mx,                                  # current, broken
        B_madz  = (mx-med)/max(mad,1e-9),              # scene-normalized peak
        C_tail  = (mx-q99)/max(q99-med,1e-9),          # tail-normalized peak
        D_clust = float(sizes.max()),                  # largest supra-thr cluster
        E_topk  = np.sort(fin)[-27:].mean()/max(med,1e-9),  # top-27 vs median
    )

def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    m = np.isfinite(s); s, y = s[m], y[m]
    if y.sum()<3 or (1-y).sum()<3: return np.nan
    r = np.argsort(np.argsort(s))+1.0
    np_, nn = y.sum(), (1-y).sum()
    return (r[y==1].sum() - np_*(np_+1)/2)/(np_*nn)

BUD = (50_000, 100_000, 250_000)
res = {b: {k: [] for k in ("A_max","B_madz","C_tail","D_clust","E_topk")} for b in BUD}
lab_ = {b: [] for b in BUD}
NS = 30
for si in range(NS):
    inv, y = scene(500+si)
    acc=None
    while acc is None or len(acc[0])<max(BUD):
        r = P.transport(*P.gen_muons(200_000), inv)
        if r is None: continue
        acc = r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
    pt, ok = P.poca(acc[0],acc[1],acc[2],acc[3])
    for b in BUD:
        lam,cnt = P.reconstruct(pt[:b], acc[5][:b], np.full(b,P.P_REF), ok[:b])
        for k,v in stats(get_field(lam,cnt)).items(): res[b][k].append(v)
        lab_[b].append(y)
    if (si+1)%10==0: print(f"  {si+1}/{NS}")

print(f"\n{'statistic':>10}" + "".join(f"{b//1000:>9}k" for b in BUD))
print("-"*40)
for k in ("A_max","B_madz","C_tail","D_clust","E_topk"):
    print(f"{k:>10}" + "".join(f"{auc(res[b][k], lab_[b]):10.3f}" for b in BUD))
print(f"\nthreat scenes: {sum(lab_[BUD[0]])}/{NS}")
