"""Cluttered conditioning sweep. No threat labels used to choose anything.
Does the air-regime angle/localisation trade survive in real cargo?"""
import numpy as np, proto_e2e as P
import poca_conditioning as PC

rng = np.random.default_rng(17)
NS, VOX, L = P.NVOX, P.VOX, P.L
X0 = {"air":304.,"steel":.01757,"water":.3608,"lead":.005612,
      "tungsten":.003504,"uranium":.003166}
BMIN,BMAX,NA = 0.08,0.28,1200

def manifest_grid(seed):
    """Metre-sized non-overlapping clutter + nested threat, rasterised for
    this diagnostic only. inv in 1/m."""
    gc = np.random.default_rng(seed*2+1); gt = np.random.default_rng(seed*2+2)
    C=np.zeros((0,3)); H=np.zeros((0,3)); mats=[]
    for _ in range(NA):
        h=(gc.random(3)*(BMAX-BMIN)+BMIN)/2
        c=gc.random(3)*(L-2*h)-(L/2-h)
        mi=int(gc.random()*2)
        if len(mats)==0 or not ((np.abs(c-C)<(h+H)).all(1)).any():
            C=np.vstack([C,c]); H=np.vstack([H,h]); mats.append(["steel","water"][mi])
    u=gt.random(4)
    tmat = ["lead","tungsten","uranium"][int(u[1]*3)] if u[0]<0.5 else None
    size = [0.06,0.08,0.10,0.14][int(u[2]*4)]
    cc=np.arange(NS)*VOX-L/2+VOX/2
    X,Y,Z=np.meshgrid(cc,cc,cc,indexing="ij"); pts=np.stack([X,Y,Z],-1)
    inv=np.full((NS,)*3,1/X0["air"]); cls=np.zeros((NS,)*3,np.int8)   # 0=air
    for c,h,m in zip(C,H,mats):
        msk=(np.abs(pts-c)<h).all(-1); inv[msk]=1/X0[m]; cls[msk]=1     # 1=benign
    if tmat is not None:
        lim=L/2-size/2-0.02; tc=gt.random(3)*(2*lim)-lim
        msk=(np.abs(pts-tc)<size/2).all(-1); inv[msk]=1/X0[tmat]; cls[msk]=2  # 2=threat
    return (inv*100/100).astype(np.float32)/100*100, cls, tmat

def zstat(q):
    """Robust z-nonuniformity: p90/p10 of per-slice density."""
    h=np.bincount(((q[:,2]+L/2)/VOX).astype(int).clip(0,NS-1),minlength=NS).astype(float)
    return np.percentile(h,90)/max(np.percentile(h,10),1e-9)

ptA=[]; ctrA=[]; thA=[]; clsA=[]
for si in range(6):
    inv, cls, tmat = manifest_grid(600+si)
    acc=None
    while acc is None or len(acc[0])<80_000:
        r=PC.transport_truth(*P.gen_muons(120_000), inv)
        acc=r if acc is None else tuple(np.concatenate([a,x]) for a,x in zip(acc,r))
    e,d,po,do,p,th,ctr = tuple(x[:80_000] for x in acc)
    pt,ok = P.poca(e,d,po,do)
    ci=np.clip(((ctr+L/2)/VOX).astype(int),0,NS-1)
    ptA.append(pt); ctrA.append(ctr); thA.append(th)
    clsA.append(cls[ci[:,0],ci[:,1],ci[:,2]])
pt=np.concatenate(ptA); ctr=np.concatenate(ctrA); th=np.concatenate(thA)
cls=np.concatenate(clsA)
inside=(np.abs(pt)<L/2).all(1)
err=np.linalg.norm(pt-ctr,axis=1)

print("median true scattering angle in cargo: %.1f mrad  (air was 0.5)"
      % (np.median(th)*1e3))
print("true centroid material: air %.2f  benign %.2f  threat %.2f\n"
      % ((cls==0).mean(),(cls==1).mean(),(cls==2).mean()))
print(f"{'th_min':>7} {'elig%':>7} {'inVol%':>7} {'medErr':>7} {'p90Err':>7} "
      f"{'zP90/P10':>9} | {'err:air':>8} {'benign':>7} {'threat':>7}")
for tmin in (0.0,0.5,1.0,2.0,5.0,10.0,20.0):
    sel = th >= tmin*1e-3
    s = sel & inside
    if s.sum()<200: print(f"{tmin:7.1f}  too few events"); continue
    row=(f"{tmin:7.1f} {100*sel.mean():7.1f} {100*s.sum()/max(sel.sum(),1):7.1f} "
         f"{np.median(err[s]):7.3f} {np.percentile(err[s],90):7.3f} "
         f"{zstat(pt[s]):9.2f} |")
    for c in (0,1,2):
        m=s&(cls==c)
        row+=f"{np.median(err[m]) if m.sum()>50 else np.nan:8.3f}"
    print(row)
