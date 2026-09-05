"""Gate 1: acceptance accounting + momentum blindness + detector smearing.
Paired design -- one fixed set of true tracks, different measurement models."""
import numpy as np, proto_e2e as P

rng = np.random.default_rng(7)

# ---- cluttered phantom (same recipe as the GPU run) ----
def clutter(seed=3, fill=0.35):
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

inv_X0, cube = clutter()

# ---- measure geometric acceptance empirically ----
N_ACC = 400_000
acc_in = acc_out = 0
store = []
while acc_out < N_ACC:
    e, d, p = P.gen_muons(200_000)
    acc_in += len(e)
    r = P.transport(e, d, p, inv_X0)
    if r is None: continue
    acc_out += len(r[0]); store.append(r)
tracks = tuple(np.concatenate([s[i] for s in store])[:N_ACC] for i in range(6))
acceptance = acc_out/acc_in
print(f"geometric acceptance: {acceptance:.4f}  ({acc_out:,}/{acc_in:,})")
print(f"incident rate 1 m^2 = 10,000/min -> accepted rate = "
      f"{10000*acceptance:,.0f}/min\n")

e_in, d_in, p_out_pt, d_out, mom, th_true = tracks

# ---- measurement models ----
def smear(entry, d, p_out_pt, d_out, sigma_m, z1, z2):
    """Refit in/out directions from two noisy hit positions per side."""
    def refit(pt, dv, za, zb):
        ta = (za - pt[:,2])/dv[:,2]; tb = (zb - pt[:,2])/dv[:,2]
        A = pt + ta[:,None]*dv; B = pt + tb[:,None]*dv
        A[:,:2] += rng.normal(0, sigma_m, (len(A),2))
        B[:,:2] += rng.normal(0, sigma_m, (len(B),2))
        nd = B - A
        nd *= np.sign((nd*dv).sum(1))[:,None]   # keep travel direction
        nd /= np.linalg.norm(nd, axis=1, keepdims=True)
        return A, nd
    Ain, din = refit(entry, d, z1, z2)              # above, going down
    Aout, dout = refit(p_out_pt, d_out, -z1, -z2)   # below
    cosang = np.clip((din*dout).sum(1), -1, 1)
    return Ain, din, Aout, dout, np.arccos(cosang)

def run(pin, din, pout, dout, theta, use_mom, label):
    pt, ok = P.poca(pin, din, pout, dout)
    rows = []
    for n in (10_000, 25_000, 50_000, 100_000, 250_000, 400_000):
        m = mom[:n] if use_mom else np.full(n, P.P_REF)
        lam, cnt = P.reconstruct(pt[:n], theta[:n], m, ok[:n])
        cnr, auc, _ = P.detect(lam, cnt, cube)
        rows.append((n, cnr, auc))
    print(f"{label:34s}" + "".join(f"{a:8.3f}" for _,_,a in rows))
    return rows

Z1, Z2 = 0.6, 1.6   # tracking planes 1.0 m apart on each side
hdr = "condition".ljust(34) + "".join(f"{n//1000:6d}k " for n in
      (10_000,25_000,50_000,100_000,250_000,400_000))
print(hdr); print("-"*len(hdr))

run(e_in, d_in, p_out_pt, d_out, th_true, True,  "A ideal (mom known, perfect det)")
run(e_in, d_in, p_out_pt, d_out, th_true, False, "B momentum-blind")
for s_mm in (0.5, 1.0):
    sm = smear(e_in, d_in, p_out_pt, d_out, s_mm/1000, Z1, Z2)
    sig_th = (s_mm/1000)*np.sqrt(2)/(Z2-Z1)*1e3
    run(sm[0], sm[1], sm[2], sm[3], sm[4], True,
        f"C det {s_mm}mm (sig_th={sig_th:.2f}mrad)")
    run(sm[0], sm[1], sm[2], sm[3], sm[4], False,
        f"D det {s_mm}mm + momentum-blind")

print(f"\nmean true scattering angle: {th_true.mean()*1e3:.2f} mrad")
print(f"median true scattering angle: {np.median(th_true)*1e3:.2f} mrad")
