# Phase 4A Protocol — Held-Out Cargo Generalization

**Status:** FROZEN BEFORE PHASE-4A RESULTS

## Question

Can the raw-event observer detect a hidden 10 cm tungsten target when the true benign cargo packing is not present anywhere in its nuisance-hypothesis library?

## Observer library

The observer retains exactly the Phase-3 nuisance library:

- cargo seeds 30000–30015

## Held-out true cargo

Every calibration/test scene gets a unique cargo packing from the same generator, with no overlap with the observer library:

- calibration benign: 40000–40199
- test benign: 50000–50199
- test threat: 60000–60199

All 600 true cargo packings are mutually distinct.

## Target / benign alternatives

- Threat: 0.10 m tungsten cube
- Benign: 50% no inserted dense object, 50% 0.10 m steel confuser
- Location: uniform over the same 27 centers x,y,z in {-0.30, 0.00, +0.30} m
- True momentum hidden
- True location hidden from blind observer

## Measurement model

Unchanged from Phase 3:

- `scene_physics.py`
- nstep = 100
- ideal incoming/outgoing track measurements
- raw angular + displacement likelihood
- 16-point momentum quadrature
- nuisance location marginalized
- static and predetermined +90° y-axis two-view schedules

## Primary budget

500 accepted tracks per schedule.

This was selected because the closed-world sweep produced strong but non-saturated static performance at 500 tracks, while 1000+ tracks were already near or at ceiling.

## Sample sizes

- calibration benign = 200
- test benign = 200
- test threat = 200

## Metrics

Primary:
- blind static AUC
- blind two-view AUC
- bootstrap 95% CI
- paired 95% CI for two-view minus static
- sensitivity at calibrated 5% scene FPR
- realized test FPR

Diagnostics:
- location-revealed AUC
- benign/threat log-Bayes-factor quantiles
- hard margin = min(threat) - max(benign)
- incident/accepted track counts

## Decision gates

### PASS
If static blind AUC >= 0.90 with useful sensitivity at 5% FPR:
- unseen same-distribution cargo preserves strong raw-event threat information
- a practical static inference branch is scientifically justified if the project continues

### GEOMETRY SIGNAL
If static AUC < 0.90, but two-view:
- improves AUC by >= 0.05
- paired 95% CI excludes 0
- improves sensitivity at the 5% FPR operating point

then at most one geometry-focused follow-up is justified.

### STOP
If both static and two-view AUC < 0.80:
- close the current target-detection project
- do not respond with a larger cargo bank, another threshold, or a learned model in this branch

### INTERMEDIATE
If results fall between these gates:
- classify as inconclusive
- because this is a time-boxed weekend research spike, default is to close unless one mechanism-level follow-up is unusually compelling

## No tuning after results

After Phase 4A:
- do not change the 500-track budget to improve the result
- do not choose different test seed ranges
- do not expand the nuisance library post hoc
- do not change target size/material
- do not tune priors or location grid

Any future study must be a separately named branch with a separately frozen protocol.

## Interpretation boundary

A pass means only:

> Under this surrogate simulator, a finite empirical nuisance library retains useful raw-event threat discrimination on unseen cargo drawn from the same cargo generator.

It does not establish robustness to distribution shift, real detector noise, Geant4 transport, arbitrary cargo, or operational false-alarm requirements.
