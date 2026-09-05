"""Is the empty-volume PoCA central peak (a) ill-conditioning, or (b) PoCA
correctly reporting the variance-weighted centroid of distributed scattering?
If (b), no angle cut flattens it and normalisation is mandatory."""
import numpy as np, proto_e2e as P

rng = np.random.default_rng(5)
NS, L = P.NVOX, P.L

def transport_truth(entry, d, p, inv_X0, nstep=100):
    """As proto_e2e.transport but ALSO returns the realised scattering
    centroid: sum(kick_i^2 * pos_i)/sum(kick_i^2)."""
    t_ex = (entry[:,2]+L/2)/(-d[:,2])
    ep = entry + t_ex[:,None]*d
    k = (np.abs(ep[:,0])<L/2)&(np.abs(ep[:,1])<L/2)
    entry,d,p,t_ex = entry[k],d[k],p[k],t_ex[k]
    n = len(p)
    s = (np.arange(nstep)+0.5)/nstep
    pts = entry[:,None,:] + (t_ex[:,None]*s)[:,:,None]*d[:,None,:]
    idx = np.clip(((pts+L/2)/P.VOX).astype(np.int32),0,NS-1)
    inv = inv_X0[idx[...,0],idx[...,1],idx[...,2]]
    Xs = inv*(t_ex/nstep)[:,None]
    logf = np.maximum(1+0.038*np.log(np.maximum(Xs.sum(1),1e-12)),0)
    ths = (13.6/p)[:,None]*np.sqrt(Xs)*logf[:,None]
    kx = rng.normal(size=(n,nstep))*ths
    ky = rng.normal(size=(n,nstep))*ths
    w = kx**2+ky**2
    centroid = (pts*w[...,None]).sum(1)/np.maximum(w.sum(1),1e-300)[:,None]
    lev = t_ex[:,None]*(1-s)
    dx,dy = (kx*lev).sum(1),(ky*lev).sum(1)
    tx,ty = kx.sum(1),ky.sum(1)
    zh=d; a=np.tile([1.,0,0],(n,1)); a[np.abs(zh[:,0])>0.9]=[0,1.,0]
    xh=np.cross(a,zh); xh/=np.linalg.norm(xh,axis=1,keepdims=True)
    yh=np.cross(zh,xh)
    do = zh+tx[:,None]*xh+ty[:,None]*yh; do/=np.linalg.norm(do,axis=1,keepdims=True)
    po = entry+t_ex[:,None]*d+dx[:,None]*xh+dy[:,None]*yh
    return entry,d,po,do,p,np.sqrt(tx**2+ty**2),centroid


if __name__ == "__main__":
    pass
    air = np.full((NS,)*3, 100.0/P.X0_CM["air"], np.float32)

    acc=None
    while acc is None or len(acc[0])<200_000:
        r = transport_truth(*P.gen_muons(150_000), air)
        acc = r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
    e,d,po,do,p,th,ctr = tuple(x[:200_000] for x in acc)
    pt, ok = P.poca(e,d,po,do)

    # --- verify den == sin^2(theta) ---
    b = (d*do).sum(1)
    print("den == sin^2(theta):  max |den - sin^2| = %.2e" %
          np.abs((1-b**2) - np.sin(np.arccos(np.clip(b,-1,1)))**2).max())

    ins = ok & (np.abs(pt)<L/2).all(1)
    cin = (np.abs(ctr)<L/2).all(1)
    print("\n--- EMPTY VOLUME, perfect detector ---")
    print("PoCA inside volume:            %.3f" % ins.mean())
    print("TRUE centroid inside volume:   %.3f" % cin.mean())
    for nm,q in (("PoCA", pt[ins]), ("TRUE centroid", ctr[cin])):
        print("  %-14s z: mean %+.4f sd %.4f | xy sd %.4f" %
              (nm, q[:,2].mean(), q[:,2].std(), q[:,:2].std()))

    def zswing(q):
        h = np.bincount(((q[:,2]+L/2)/P.VOX).astype(int).clip(0,NS-1), minlength=NS)
        return h.max()/max(h[h>0].min(),1)
    print("  z-density swing   PoCA %.2fx   TRUE centroid %.2fx"
          % (zswing(pt[ins]), zswing(ctr[cin])))

    # --- localisation error vs angle, and the theta_min sweep ---
    both = ins & cin
    err = np.linalg.norm(pt[both]-ctr[both], axis=1)
    tb = th[both]
    print("\n--- PoCA localisation error vs scattering angle (EMPTY) ---")
    qs = np.percentile(tb, [0,20,40,60,80,95,100])
    print(f"{'theta band (mrad)':>22} {'n':>7} {'median err (m)':>15}")
    for i in range(len(qs)-1):
        m = (tb>=qs[i])&(tb<qs[i+1] if i<len(qs)-2 else tb<=qs[i+1])
        if m.sum(): print(f"{qs[i]*1e3:9.3f}-{qs[i+1]*1e3:<11.3f} {m.sum():7d} "
                          f"{np.median(err[m]):15.3f}")

    print("\n--- theta_min sweep (EMPTY volume) ---")
    print(f"{'theta_min (mrad)':>17} {'eligible%':>10} {'z swing':>9} {'med err':>9}")
    for tmin in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        sel = ins & (th >= tmin*1e-3)
        s2 = both & (th >= tmin*1e-3)
        print(f"{tmin:17.2f} {100*sel.mean():10.1f} {zswing(pt[sel]):9.2f} "
              f"{np.median(np.linalg.norm(pt[s2]-ctr[s2],axis=1)) if s2.sum() else np.nan:9.3f}")
