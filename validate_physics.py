"""NumPy validation of the muon MCS forward model + PoCA. Physics check only."""
import numpy as np

rng = np.random.default_rng(0)

# Radiation lengths (cm)
X0 = {"air": 30400.0, "steel": 1.757, "tungsten": 0.3504, "water": 36.08}

def highland(p_MeV, X):
    """RMS plane scattering angle (rad). X = thickness in radiation lengths."""
    X = np.maximum(X, 1e-12)
    return (13.6 / p_MeV) * np.sqrt(X) * (1.0 + 0.038 * np.log(X))

# --- Check 1: known-case sanity ---
print("Highland spot checks (3 GeV muon):")
for mat, L in [("tungsten", 10.0), ("steel", 10.0), ("air", 100.0)]:
    X = L / X0[mat]
    th = highland(3000.0, X)
    print(f"  {L:5.0f} cm {mat:9s}  X={X:10.3e}  theta0={th*1e3:9.4f} mrad")

# --- Check 2: cos^2(theta) zenith sampling ---
def sample_zenith(n, th_max=np.pi/3):
    # pdf ~ cos^2(th) sin(th) -> CDF ~ (1-cos^3)/(1-cos^3_max)
    u = rng.random(n)
    c = np.cbrt(1.0 - u * (1.0 - np.cos(th_max)**3))
    return np.arccos(c)

th = sample_zenith(200000)
print(f"\nZenith: mean={np.degrees(th.mean()):.2f} deg, "
      f"frac<30deg={np.mean(th < np.pi/6):.3f}")

# --- Check 3: momentum spectrum ~ p^-2.7 above 1 GeV ---
def sample_p(n, pmin=1000.0, alpha=2.7):
    u = rng.random(n)
    return pmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))

p = sample_p(200000)
print(f"Momentum: mean={p.mean()/1000:.2f} GeV, median={np.median(p)/1000:.2f} GeV")

# --- Check 4: multi-step scattering vs single-step equivalence ---
# Variance must add: sum of per-step theta0^2 == total theta0^2 (ignoring log term)
n_steps = 100
X_total = 10.0 / X0["tungsten"]
X_step = X_total / n_steps
logf = 1.0 + 0.038 * np.log(X_total)
th_step = (13.6 / 3000.0) * np.sqrt(X_step) * logf
th_tot_from_steps = np.sqrt(n_steps * th_step**2)
th_tot_direct = highland(3000.0, X_total)
print(f"\nVariance additivity: stepwise={th_tot_from_steps*1e3:.4f} mrad, "
      f"direct={th_tot_direct*1e3:.4f} mrad, "
      f"ratio={th_tot_from_steps/th_tot_direct:.6f}")

# --- Check 5: PoCA recovers a known scattering point ---
def poca(p1, d1, p2, d2):
    w0 = p1 - p2
    a = np.sum(d1*d1, -1); b = np.sum(d1*d2, -1); c = np.sum(d2*d2, -1)
    d = np.sum(d1*w0, -1); e = np.sum(d2*w0, -1)
    den = a*c - b*b
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    sc = (b*e - c*d) / den
    tc = (a*e - b*d) / den
    return 0.5 * ((p1 + sc[..., None]*d1) + (p2 + tc[..., None]*d2))

# muon down the z axis, kinked at z = +0.2
true_pt = np.array([[0.05, -0.03, 0.2]])
d_in = np.array([[0.1, 0.05, -1.0]]); d_in /= np.linalg.norm(d_in, axis=-1, keepdims=True)
d_out = d_in + np.array([[0.02, -0.015, 0.0]]); d_out /= np.linalg.norm(d_out, axis=-1, keepdims=True)
p_in = true_pt - 3.0 * d_in
p_out = true_pt + 3.0 * d_out
rec = poca(p_in, d_in, p_out, d_out)
print(f"\nPoCA recovery: true={true_pt[0]}, rec={rec[0]}, "
      f"err={np.linalg.norm(rec-true_pt):.2e} m")

# --- Check 6: flux normalization ---
# ~1 muon / cm^2 / min through a horizontal plane at sea level
for area_m2, label in [(1.0, "1 m^2 spike volume"), (14.4, "20ft container top")]:
    rate = area_m2 * 1e4  # muons per minute
    print(f"Flux: {label:22s} -> {rate:9.0f} muons/min; "
          f"1M tracks = {1e6/rate:6.1f} min")
