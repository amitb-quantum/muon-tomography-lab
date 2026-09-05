"""Does the tight seed convergence at 250k reflect a converged measurement,
or convergence to ONE cargo packing? Vary the phantom, not the muons."""
import numpy as np, proto_e2e as P

def clutter(seed, fill=0.35):
    r = np.random.default_rng(seed)
    inv = np.full((P.NVOX,)*3, 100.0/P.X0_CM["air"], np.float32)
    n, placed = P.NVOX, 0
    while placed < fill*n**3:
        sz = r.integers(4,14,3); o = r.integers(0,n-sz,3)
        mat = "steel" if r.random() < 0.45 else "water"
        inv[o[0]:o[0]+sz[0], o[1]:o[1]+sz[1], o[2]:o[2]+sz[2]] = 100.0/P.X0_CM.get(mat,36.08)
        placed += int(np.prod(sz))
    c = np.arange(n)*P.VOX - P.L/2 + P.VOX/2
    X,Y,Z = np.meshgrid(c,c,c,indexing="ij")
    cube = (np.abs(X)<P.CUBE/2)&(np.abs(Y)<P.CUBE/2)&(np.abs(Z)<P.CUBE/2)
    inv[cube] = 100.0/P.X0_CM["tungsten"]
    return inv, cube

N = 250_000
print(f"{'scene':>6} {'non-air%':>9} {'AUC@100k':>10} {'AUC@250k':>10}")
res100, res250 = [], []
for scene in range(10):
    inv, cube = clutter(seed=100+scene)
    acc = None
    while acc is None or len(acc[0]) < N:
        r = P.transport(*P.gen_muons(200_000), inv)
        if r is None: continue
        acc = r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
    a = tuple(x[:N] for x in acc)
    pt, ok = P.poca(a[0],a[1],a[2],a[3])
    out = []
    for n in (100_000, N):
        lam, cnt = P.reconstruct(pt[:n], a[5][:n], np.full(n, P.P_REF), ok[:n])
        out.append(P.detect(lam, cnt, cube)[1])
    res100.append(out[0]); res250.append(out[1])
    print(f"{scene:6d} {(inv>1.0).mean()*100:9.1f} {out[0]:10.3f} {out[1]:10.3f}")

for lab, r in (("100k", res100), ("250k", res250)):
    r = np.array(r)
    print(f"\n{lab}: mean {r.mean():.3f}  sd {r.std(ddof=1):.3f}  "
          f"range [{r.min():.3f}, {r.max():.3f}]  spread {r.max()-r.min():.3f}")
