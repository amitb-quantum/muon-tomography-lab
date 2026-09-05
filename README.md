# Muon Tomography Lab

GPU-accelerated experiments in cosmic-ray muon scattering tomography, with a focus on a deceptively hard question:

> **Can a detector find high-Z material inside heterogeneous cargo when it does not already know where to look?**

This repository started as a fast physics/reconstruction spike and evolved into a small falsification-driven study of Point of Closest Approach (PoCA) tomography. The most important result is not a new detector. It is a boundary:

**the local scattering signal is strong, but blind spatial search with PoCA throws away much of that signal.**

At 250,000 accepted tracks, a ground-truth "oracle" that knows where the target is reaches an AUC of about **0.97**, while the best calibrated blind scene-level search in this repo reaches about **0.72**.

That gap is the current research problem.

---

## In plain language

Cosmic rays constantly create muons in the atmosphere. Muons pass through ordinary material easily, but dense materials such as lead, tungsten, and uranium deflect them more strongly.

A muon tomography system measures a muon's incoming and outgoing tracks and tries to infer where inside an object the scattering happened.

The initial idea was simple:

1. simulate cosmic-ray muons passing through cluttered cargo,
2. reconstruct their scattering locations,
3. identify unusually strong scattering,
4. ask whether more intelligent acquisition could reduce scan time.

The first results looked excellent. A centered 10 cm tungsten target was easy to separate from its background.

Then the experiment was made progressively harder and more realistic.

Several attractive intermediate results disappeared.

That was useful.

The final surviving result is that **high-Z scattering is detectable locally, but PoCA-based blind localization/search is the bottleneck in heterogeneous cargo**.

---

## What is in this repository

The code covers several stages of the investigation:

- GPU-vectorized cosmic-ray muon generation and multiple-Coulomb-scattering transport
- cluttered and simple cargo phantoms
- momentum-aware and momentum-blind reconstruction
- detector-position smearing
- PoCA reconstruction
- continuous-coordinate scene manifests
- resolution-independent forward simulation
- immutable simulated event records
- instrument-response calibration
- scene-level threat/benign evaluation
- occupancy-only tests
- scattering-magnitude (`lambda`) tests
- blind-search diagnostics
- spatially conditioned empirical tail calibration

The work was developed and tested on an **NVIDIA GeForce RTX 5090** with PyTorch/CUDA.

Generated `.npz` experiment outputs are intentionally ignored by Git.

---

## Research progression

### 1. Initial spike

The first experiment placed a 10 cm tungsten cube inside a 1 m³ cluttered volume and reconstructed scattering with PoCA.

The initial voxel-level results were strong and increased monotonically with track count.

For example, under the first idealized model:

| Accepted tracks | Approx. exposure | Voxel AUC |
|---:|---:|---:|
| 10,000 | 2.8 min | 0.847 |
| 50,000 | 14.2 min | 0.909 |
| 100,000 | 28.3 min | 0.934 |
| 250,000 | 70.8 min | 0.970 |
| 500,000 | 141.7 min | 0.990 |

The exposure numbers include the measured geometric acceptance of about 0.353 for the modeled 1 m² geometry.

This result established that the simulator/reconstruction pipeline could recover a strong high-Z signal.

It did **not** establish operational scene detection.

---

### 2. Remove privileged assumptions

The spike was then made less favorable:

- exact per-track momentum removed,
- detector-position noise added,
- multiple random muon realizations tested,
- cluttered cargo retained.

The signal survived.

At 250,000 tracks, a momentum-blind reconstruction with 1 mm detector smearing still produced a voxel AUC around **0.95** for the original fixed scene.

But a larger experiment showed that this apparent precision was misleading: variation between cargo scenes was far larger than variation between repeated muon draws for one scene.

That forced the evaluation to move from voxel-level scoring in one known scene to **scene-level threat detection across many independently generated cargo packings**.

---

### 3. Separate the physical world from the reconstruction grid

An early resolution sweep exposed a serious confound: changing voxel resolution also changed the generated cargo geometry and the forward scattering calculation.

The simulator was rebuilt so that:

- scenes are generated in continuous physical coordinates,
- forward physics no longer depends on reconstruction voxel size,
- identical simulated muon events can be reconstructed at different resolutions,
- manifest and event hashes provide regression checks.

A 400-scene invariance run completed with **0 violations**.

This separation is foundational: reconstruction resolution now changes only the representation, not the simulated world.

---

### 4. PoCA has a strong instrument response even in empty space

An empty-volume diagnostic found that PoCA locations are not spatially uniform even when there is essentially no material present.

With pure-air transport, PoCA positions were strongly concentrated toward the center of the volume.

This effect persisted across exposure budgets and produced roughly a **6× z-slice p90/p10 density variation** in the final Stage-A calibration.

A simple minimum scattering-angle cut did not solve the problem. In cargo, realistic cuts discarded many events while providing little localization improvement.

The final pipeline therefore fixes:

```text
theta_min = 0
```

and treats the empty-volume PoCA pattern as an **instrument response that must be calibrated**, not as physical cargo structure.

---

## What was tested and what survived

### Occupancy as a detector: rejected

One hypothesis was that high-Z material might create an independently useful concentration of PoCA points.

After correcting for:

1. the empty-manifest PoCA instrument response, and
2. the normal structure induced by benign cargo,

occupancy-only scene detection collapsed to chance.

On 400 development scenes:

| Tracks | Occupancy-only AUC |
|---:|---:|
| 25,000 | 0.509 |
| 50,000 | 0.555 |
| 100,000 | 0.524 |
| 250,000 | 0.544 |

The outside-volume PoCA fraction was also non-informative at approximately **0.49–0.50 AUC**.

**Conclusion:** occupancy is not a useful standalone threat channel in this model after proper normalization.

---

### Scattering magnitude: survives

The scattering-magnitude channel was tested separately, without using PoCA occupancy as a hidden count gate.

A blind global scene score was only moderate:

| Tracks | Raw global AUC |
|---:|---:|
| 25,000 | 0.593 |
| 50,000 | 0.689 |
| 100,000 | 0.609 |
| 250,000 | 0.592 |

This non-monotonic behavior initially looked like weak physics.

It was not.

---

## The key finding: local signal is strong, search is weak

A diagnostic compared the blind global search with an **oracle** that uses the exact same reconstructed scattering field but is allowed to inspect the known target region.

| Tracks | Blind global AUC | Oracle AUC |
|---:|---:|---:|
| 25,000 | 0.593 | 0.664 |
| 50,000 | 0.689 | 0.885 |
| 100,000 | 0.609 | 0.966 |
| 250,000 | 0.592 | 0.971 |

This is the central result of the project.

By 250,000 tracks, the underlying scattering field contains enough information for approximately **0.97 AUC** if the correct region is known.

The blind PoCA search fails because the global maximum increasingly selects badly behaved, weakly illuminated peripheral voxels rather than the target.

At 250,000 tracks, the uncalibrated argmax had migrated to a median distance of only about **1 cm from the volume boundary**.

---

## One diagnosis-derived correction

The final permitted correction was a conditionally pooled empirical tail calibration.

Voxels were stratified using instrument properties only:

- Stage-A expected PoCA response `G(v, N)`
- distance from the volume edge

The filtered scattering value at each voxel was converted into an empirical benign-tail probability within its stratum before taking the global maximum.

This partially fixed the mechanism.

| Tracks | Raw AUC | Calibrated AUC | Oracle AUC |
|---:|---:|---:|---:|
| 25,000 | 0.593 | 0.560 | 0.666 |
| 50,000 | 0.689 | 0.732 | 0.884 |
| 100,000 | 0.609 | 0.723 | 0.964 |
| 250,000 | 0.592 | 0.720 | 0.967 |

The pathological march to the wall improved substantially:

```text
uncalibrated edge@argmax:
0.190 -> 0.130 -> 0.050 -> 0.010 m

calibrated edge@argmax:
0.210 -> 0.170 -> 0.110 -> 0.110 m
```

So the diagnosis was real and the correction helped.

It did not close the gap.

At 250,000 tracks:

```text
oracle AUC      ~0.97
blind calibrated AUC ~0.72
```

The empirical calibration also saturated for roughly one-third of scenes, indicating that further improvement would require a substantially different calibration/search approach rather than another minor threshold adjustment.

---

## What the project discovered

### Positive findings

- High-Z material produces a strong local scattering signal in the surrogate physics model.
- The signal survives removal of exact momentum information.
- Moderate detector-position smearing does not destroy the basic signal.
- Cargo-scene geometry matters much more than repeated muon noise once exposure becomes substantial.
- Target regions with better instrument coverage are much easier to detect.
- The dominant late-stage failure is spatial localization/search, not lack of local scattering contrast.

### Negative findings

- PoCA occupancy is not an independent threat detector after proper instrument and benign-cargo normalization.
- Outside-volume PoCA rate is not useful as a standalone threat signal.
- Minimum scattering-angle cuts do not provide an attractive conditioning/event-retention trade in cargo.
- A naive global maximum over PoCA-derived scattering fields is dominated by spatially heterogeneous extreme-value behavior.
- More muons alone do not solve the blind-search problem under the current PoCA detector.

---

## Why the project stops here

This repository intentionally stops rather than continuing to tune the same detector.

The final stop rule was defined before the last calibration run:

> If calibrated blind detection remains only moderate, remains far below the oracle, or requires another sequence of fitted search corrections, stop the PoCA branch.

That condition was met.

Continuing with more thresholds, more strata, more kernels, or a neural network on top of the same search would turn the project into post-hoc optimization.

The current result is therefore preserved as a boundary finding:

> **PoCA carries strong local high-Z information, but under heterogeneous unknown cargo, scene-level detection is limited by blind spatial localization, coverage, and extreme-value structure in the search.**

---

## What comes next

A future continuation should **not** be `phase2e` with another PoCA threshold.

It should start from a different reconstruction/localization paradigm.

Candidate directions include:

- MLEM-style reconstruction
- Bayesian spatial inference
- neural-field / implicit reconstruction
- learned spatial posteriors
- active multi-view acquisition driven by coverage and uncertainty

The active-acquisition idea remains interesting, but the results here change its motivation.

The original idea was:

> use fewer muons by deciding when enough evidence has arrived.

The stronger problem exposed here is:

> **the physics contains information, but some spatial regions are poorly localized and poorly searched. Can the next measurement geometry be chosen to reduce that spatial uncertainty?**

Before making physical or operational claims, the fast surrogate transport model should also be validated against **Geant4** or equivalent higher-fidelity particle transport.

---

## Important limitations

This is a research spike, not a validated radiation-imaging system.

Current limitations include:

- fast surrogate multiple-Coulomb-scattering transport rather than Geant4
- simplified cosmic-ray spectrum
- simplified detector model
- no detector efficiency losses
- no detector-material scattering
- no absorption/stopping modality
- simulated cargo distributions rather than measured cargo data
- PoCA reconstruction only for the completed branch
- development-set results only; no claim of external generalization

The initial transport implementation was checked against the Highland multiple-scattering formula and analytic PoCA cases, but that is not a substitute for full particle-transport validation.

---

## Repository map

Key scripts include:

- `muon_spike.py` — original GPU forward-model and PoCA spike
- `scene_physics.py` — continuous physical manifests and resolution-independent forward model
- `scene_harness.py` — multi-scene evaluation harness
- `poca_conditioning.py` — PoCA conditioning diagnostics
- `clutter_conditioning.py` — conditioning tests in clutter
- `phase2_occupancy.py` — occupancy-only Stage-A/Stage-B experiment
- `phase2b_lambda.py` — scattering-magnitude-only experiment
- `phase2c_argmax.py` — oracle/global-search and argmax migration diagnosis
- `phase2d_calibrated.py` — final conditionally calibrated spatial-search experiment

Some patch and diagnostic scripts are retained intentionally as provenance of the experimental hardening process.

See [`FINDINGS.md`](FINDINGS.md) for the condensed result ledger.

---

## Status

**Current state: PoCA branch frozen.**

The repository is preserved as an experimental record and as a starting point for a future reconstruction/localization branch.

The important result is not that muon tomography failed.

It is that, in this experiment, **the signal exists locally and the search loses it**.
