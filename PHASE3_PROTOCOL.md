# Phase 3 Protocol — Raw-Event Ideal Observer

**Status:** FROZEN BEFORE PHASE-3 RESULTS  
**Repository:** `amitb-quantum/muon-tomography-lab`

## 1. Question

Phase 2 established that the current PoCA-based scene detector performs much worse when the target location is unknown than when a scoring region is supplied. That gap is **not** itself a measurement of information discarded by PoCA, because the location-revealed oracle solves an easier statistical problem.

Phase 3 asks the missing question directly:

> **How much blind threat-vs-benign information is present in the raw incoming/outgoing muon measurements under the current simulator, before choosing another reconstruction method or an active acquisition strategy?**

The experiment must distinguish three possibilities:

1. the current static measurements already contain strong blind-discrimination information and PoCA/search is the main limitation;
2. a complementary acquisition geometry adds substantial information unavailable to the current static view;
3. even an ideal observer performs poorly, so more elaborate reconstruction cannot rescue this sensing problem at the declared budget/model.

This phase is an **information-content experiment**, not a claim about operational cargo screening.

## 2. Corrections carried forward from the Phase-2 audit

The following interpretations are retired for Phase 3:

- The oracle–blind AUC gap is **not** treated as recoverable information loss.
- Stage-A PoCA endpoint density `G(v,N)` is **not** called physical illumination or physical ray coverage.
- The prior `G_T` split is **not** used as evidence for active view selection because it was confounded by target size.
- The custom Phase-2 AUC implementation is not reused; Phase 3 uses tie-correct AUC.
- Shared source/scattering streams across all scenes are not reused. Every realization receives fresh deterministic source and scattering seeds.

The Phase-3 observer never receives true per-track momentum.

## 3. Frozen challenge distribution

### 3.1 Volume and simulator

- Volume: 1 m × 1 m × 1 m.
- Forward simulator: `scene_physics.py`.
- Physics discretization: `nstep = 100`.
- Detector model: the current **ideal detector** event record from `scene_physics.py`.
- No detector smearing is added in this phase. This is deliberate: Phase 3 first estimates an information ceiling under the existing surrogate model.
- Forward acceptance remains whatever `scene_physics.simulate()` implements. Limitations of that acceptance model remain explicit.

### 3.2 Benign cargo library

Use 16 fixed clutter packings generated from the clutter stream of:

```text
30000, 30001, ..., 30015
```

Any target/confuser originally drawn by `make_manifest()` is stripped. Only the clutter boxes are retained.

These 16 packings form the finite nuisance library for the unknown-cargo observer.

### 3.3 Target

Threat target:

```text
material = tungsten
size     = 0.10 m cube
```

Target size is fixed for all Phase-3 scenes. No mixed-size ladder is allowed in the gate.

### 3.4 Benign alternatives

The benign class is a 50/50 mixture of:

1. no inserted dense object;
2. a 0.10 m **steel** cube at a hidden candidate location.

The steel cube is a same-size dense confuser. The target/confuser overrides underlying clutter where boxes overlap, matching the target-last material rule in the current forward simulator.

### 3.5 Candidate locations

Use the 27 fixed centers on the Cartesian grid:

```text
x, y, z ∈ {-0.30, 0.00, +0.30} m
```

All locations are declared before results.

Threat scenes draw one of the 27 locations uniformly.

Benign scenes also draw a hidden **probe location** uniformly. If the benign subtype is steel, steel is inserted at that location. If the subtype is no-target, the probe remains latent and is used only for the location-revealed diagnostic so that the diagnostic is defined symmetrically across classes.

### 3.6 Class and nuisance priors used by the observer

Threat hypothesis:

```text
P(clutter)  = uniform
P(location) = uniform
material    = tungsten
```

Benign hypothesis:

```text
P(no target)        = 0.5
P(steel confuser)   = 0.5
P(clutter)          = uniform
P(location | steel) = uniform
```

These priors are part of the frozen finite challenge. They are not claims about real-world cargo frequencies.

## 4. Acquisition schedules

Phase 3 compares two **predetermined** schedules at equal accepted-track budget.

### A. Static

All accepted tracks are generated with the original cargo orientation.

### B. Complementary two-view

Half the accepted tracks use the original orientation.

Half use a +90° rotation of the cargo about the y-axis:

```text
(x, y, z) -> (z, y, -x)
```

Because the current scene representation uses axis-aligned boxes, a 90° rotation preserves the AABB representation by permuting center coordinates and half-sizes.

This is an **ideal complementary-view diagnostic**. It is not to be described as physically equivalent to tilting a real detector. The current cosmic-flux and aperture model is not sufficient for that hardware claim.

For every scene and schedule, source/scattering seeds are unique to that realization. Reproducibility comes from deterministic seed derivation, not shared measurement noise across scenes.

## 5. Primary budget and sample sizes

Primary gate:

```text
accepted tracks per schedule = 50,000
```

For the two-view schedule:

```text
25,000 original + 25,000 complementary
```

Primary sample sizes:

```text
calibration benign = 200
test benign        = 200
test threat        = 200
```

The calibration benign set is used only to set the operating threshold for 5% scene false-positive rate. It is not used to fit the ideal-observer likelihood.

A `--smoke` mode in the implementation may use a much smaller library, track count, and scene count for code validation. **Smoke results are non-scientific and must never be reported as Phase-3 evidence.**

## 6. Observer

### 6.1 Measurements available to the observer

For every accepted track, the observer may use:

- incoming position;
- incoming direction;
- outgoing position;
- outgoing direction.

The observer may **not** use:

- true simulated momentum `ev["p"]`;
- true scattering kicks;
- true material labels;
- true target location, except in the explicitly labeled location-revealed diagnostic.

### 6.2 Raw residuals

From the event record, recover in the same local transverse basis used by `scene_physics.forward()`:

- angular deflections `tx`, `ty`;
- transverse displacements `dx`, `dy`.

For a candidate material hypothesis, use the simulator's undeflected incoming ray and the same `nstep=100` midpoint discretization to compute:

\[
J_0 = \sum_j \lambda_j \Delta s
\]

\[
J_1 = \sum_j \lambda_j \Delta s (L-s_j)
\]

\[
J_2 = \sum_j \lambda_j \Delta s (L-s_j)^2
\]

where \(\lambda_j = 1/X_0\) at the path midpoint.

For one transverse projection, the covariance conditional on momentum \(p\) is

\[
\Sigma(p)
=
\left(\frac{13.6}{p}\right)^2
\left[1+0.038\log J_0\right]^2
\begin{pmatrix}
J_0 & J_1 \\
J_1 & J_2
\end{pmatrix}.
\]

The two transverse projections are conditionally independent given the **same latent momentum**.

### 6.3 Momentum marginalization

True event momentum is hidden.

The observer integrates over the simulator's momentum law generated by:

```python
p = 1000 * (1-u)**(-1/1.7),  u ~ Uniform(0,1)
```

Numerical integration uses deterministic Gauss-Legendre quadrature in the latent uniform variable.

### 6.4 Blind Bayes factor

For observed data \(D\),

\[
\Lambda(D)
=
\frac{\sum_{h\in H_1}\pi_h p(D\mid h)}
     {\sum_{h\in H_0}\pi_h p(D\mid h)}.
\]

The reported scene score is `log Λ`.

This score marginalizes target location rather than maximizing over it, so the unknown-location search penalty is part of the statistical problem rather than treated as a post-hoc multiple-comparison correction.

### 6.5 Location-revealed diagnostic

A second score is computed with the scene's predeclared probe location supplied to the observer.

It still hides momentum and, in the unknown-cargo phase, still marginalizes cargo.

This score is a diagnostic for the cost of unknown location. It is not an operational detector.

## 7. Staged computation

### Phase 3A — known cargo, unknown target location

The observer is told which of the 16 benign clutter packings generated the scene.

It is **not** told:

- threat/benign label;
- target/confuser subtype;
- target location;
- momentum.

This favorable test isolates the unknown-location/search problem from unknown-cargo ambiguity and is computationally much cheaper.

Run Phase 3A first.

### Phase 3B — unknown cargo and unknown target location

Run only if Phase 3A justifies continuation.

The observer marginalizes over all 16 clutter packings as well as location and benign subtype.

Phase 3B is the finite-library version of the actual blind unknown-cargo question.

## 8. Metrics

For each schedule and observer mode report:

- tie-correct scene AUC;
- bootstrap 95% CI for AUC;
- paired bootstrap 95% CI for the two-view minus static AUC difference;
- sensitivity at a threshold calibrated to 5% scene FPR;
- realized test-set FPR at that threshold;
- location-revealed AUC as a diagnostic;
- accepted and incident track counts.

AUC is not the sole gate. The 5% FPR operating point must also be inspected.

## 9. Go / no-go rules

These are research spending gates, not operational security requirements.

### Gate A — static information is already strong

If the **blind static ideal observer** reaches:

```text
AUC >= 0.90
```

and shows useful sensitivity at 5% scene FPR, then the existing static measurements contain strong blind-discrimination information.

**Decision:** continue with better static inference/reconstruction. Do not pursue active geometry yet.

### Gate B — complementary geometry adds decisive information

If static blind ideal-observer AUC is below 0.90, but the predetermined two-view schedule:

```text
reaches AUC >= 0.90
AND
improves AUC by >= 0.10
AND
the 95% CI for the improvement excludes 0
```

with a meaningful improvement at the 5% FPR operating point:

**Decision:** continue acquisition-geometry research.

This establishes value for a complementary view. It does **not** yet establish value for an adaptive policy.

### Gate C — favorable ideal observer is weak

If, in Phase 3A, both static and two-view ideal observers remain:

```text
AUC < 0.80
```

under the ideal detector and declared budget:

**Decision:** stop this target-detection concept at this budget/model. A more elaborate PoCA/MLEM/neural reconstruction cannot outperform the declared-model ideal observer.

### Intermediate region

Results between these gates are **inconclusive**.

Do not convert an inconclusive result into a new series of score tweaks.

## 10. What Phase 3 does not prove

Even a clean Phase-3 pass remains a surrogate-model result.

It does not validate:

- the cosmic spectrum;
- post-scattering aperture acceptance;
- detector material;
- detector efficiency;
- detector spatial/angular resolution;
- momentum response;
- absorption/stopping;
- arbitrary real cargo;
- operational false-alarm rates.

If Phase 3 passes a continuation gate, the next physics gate is comparison against Geant4 or equivalent higher-fidelity particle transport before external performance claims.

## 11. Frozen interpretation

The purpose of Phase 3 is not to rescue the project.

It is to answer one question cleanly:

> **Does the current simulated measurement stream already contain strong blind threat information, or does a complementary geometry materially increase the ideal observer's information?**

Only after that answer should the project choose among:

- better static reconstruction/inference;
- acquisition geometry research;
- stopping.
