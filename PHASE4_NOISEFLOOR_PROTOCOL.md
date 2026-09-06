# Phase 4A-NF Protocol — Likelihood Noise-Floor Sensitivity

**Status:** FROZEN BEFORE RESULTS

## Purpose
Phase 4A held-out-cargo detection collapsed to chance, but its Gaussian raw-event likelihood had no irreducible measurement/model-error floor. This diagnostic tests whether that collapse persists when the likelihood is regularized so small cargo-model errors cannot create arbitrarily sharp Gaussian penalties.

This is a **likelihood-floor sensitivity test**, not a validated detector model. The simulated event stream remains ideal/noiseless.

## Frozen covariance floor
For each transverse projection, replace `Sigma_phys(p)=s(p)K` by

```
Sigma(p) = [[s*k00 + sigma_theta^2, s*k01],
            [s*k01, s*k11 + sigma_x^2]]
```

with:

```
sigma_theta = 1.0 mrad
sigma_x     = 1.0 mm
```

Because `s=(13.6/p)^2` depends on latent momentum, these floors are added inside the momentum quadrature, after the Highland scale.

## Scene distribution
Exactly Phase 4A:
- observer cargo library: seeds 30000–30015
- calibration true cargo: 40000–40199
- test benign true cargo: 50000–50199
- test threat true cargo: 60000–60199
- all true cargo manifests distinct and absent from the observer library
- 10 cm tungsten threat
- 50/50 no-target vs 10 cm steel benign alternative
- same 27 locations
- momentum hidden
- static vs predetermined two-view

## Primary budget
500 accepted tracks per schedule, matching Phase 4A exactly.

## Decision rule
- **COLLAPSE PERSISTS:** both blind AUCs < 0.60 → zero-floor confound is insufficient; close project.
- **MATERIAL RECOVERY:** either blind AUC >= 0.70 → likelihood floor materially changes the conclusion; run exactly one 10,000-track check with the same floors, then close.
- **INTERMEDIATE:** best blind AUC in [0.60,0.70) → report sensitivity to likelihood regularization and close; no tuning branch.

No other sigma values, priors, target sizes, cargo libraries, or budgets are to be tried today.

## Interpretation boundary
This test does not simulate physical detector resolution. A real detector study must perturb measured tracks consistently and derive angular/position covariance from detector geometry.
