# Findings

This file records the main experimental conclusions of `muon-tomography-lab`.

It is intentionally more concise than the development history in the source code and focuses on findings that survived later controls.

---

## 1. Initial local detection works

A fast GPU surrogate model generated cosmic-ray muons, transported them through clutter using multiple-Coulomb-scattering physics, and reconstructed scattering with Point of Closest Approach (PoCA).

A centered 10 cm tungsten target produced strong voxel-level separation from surrounding clutter.

At 250,000 accepted tracks, the original fixed-scene voxel AUC was about 0.97.

This demonstrated a recoverable local high-Z signal.

It did not demonstrate scene-level detection.

---

## 2. Exposure accounting had to be corrected

The first scan-time calculation used the top-plane incident flux while the reconstruction retained only through-going tracks.

The modeled geometry accepts about 35.3% of incident muons.

The reported accepted-track rate was corrected to roughly:

```text
3,530 accepted tracks/min
```

for the modeled 1 m² geometry.

---

## 3. Exact momentum is helpful but not essential

Removing exact per-track momentum degraded low-exposure results but did not eliminate the high-Z signal.

The project therefore continued in momentum-blind mode rather than relying on privileged momentum information.

---

## 4. Scene variation dominates repeated-muon variation

Repeated muon draws through one fixed cargo scene produced very tight high-budget results.

Changing the cargo packing produced much larger variation.

A preliminary 10-scene experiment showed that scene difficulty was also persistent across exposure levels, motivating later interest in adaptive acquisition.

The key lesson was methodological:

> repeated simulation seeds for one scene do not establish robustness across cargo scenes.

---

## 5. Simulation and reconstruction resolution were initially coupled

An early voxel-size experiment was invalid because changing `nvox` changed:

- the physical clutter distribution,
- RNG consumption,
- and the forward scattering calculation.

The simulator was rebuilt around continuous-coordinate physical manifests and resolution-independent forward physics.

A 400-scene invariance suite completed with zero violations.

This is the architecture used by the later experiments.

---

## 6. PoCA has a large empty-volume spatial response

Even in pure air, PoCA positions are highly nonuniform.

The final Stage-A calibration showed a stable z-slice p90/p10 occupancy ratio of roughly 6 across budgets.

This is an instrument/reconstruction response, not cargo.

A minimum scattering-angle cut did not remove the problem and was not worth its event loss in cluttered cargo.

Final Phase-2 experiments therefore use:

```text
theta_min = 0
```

and calibrate the instrument response explicitly.

---

## 7. Occupancy-only detection is negative

After:

1. empty-manifest instrument calibration, and
2. cross-fitted benign-cargo normalization,

PoCA occupancy did not discriminate threat from benign scenes.

400-scene DEV result:

| Accepted tracks | AUC |
|---:|---:|
| 25,000 | 0.509 |
| 50,000 | 0.555 |
| 100,000 | 0.524 |
| 250,000 | 0.544 |

The outside-volume PoCA fraction was also approximately chance:

```text
AUC ~= 0.49-0.50
```

Conclusion:

> apparent occupancy gains seen earlier were produced by support/instrument artifacts, not a useful independent threat channel.

---

## 8. Scattering magnitude survives, but naive blind search does not

A scattering-magnitude-only detector retained above-chance information, but the global maximum behaved non-monotonically:

| Accepted tracks | Raw global AUC |
|---:|---:|
| 25,000 | 0.593 |
| 50,000 | 0.689 |
| 100,000 | 0.609 |
| 250,000 | 0.592 |

A diagnostic then evaluated the same field at the known target region.

The oracle result was:

| Accepted tracks | Oracle AUC |
|---:|---:|
| 25,000 | 0.664 |
| 50,000 | 0.885 |
| 100,000 | 0.966 |
| 250,000 | 0.971 |

This is the key result.

The local scattering signal becomes very strong with exposure.

The blind global search does not recover it.

---

## 9. Why the blind search fails

As exposure increased, the valid spatial search region expanded.

The uncalibrated global maximum increasingly selected weakly illuminated peripheral voxels.

Median distance of the winning voxel from the box boundary:

```text
25k   0.190 m
50k   0.130 m
100k  0.050 m
250k  0.010 m
```

The benign extreme-value distribution also grew with the search volume.

The failure was therefore not simply weak scattering physics.

It was a spatial-search calibration problem.

---

## 10. Conditional calibration partially fixes the mechanism

The final permitted correction grouped voxels using instrument-only properties:

- Stage-A expected response `G(v,N)`
- edge distance

Within each stratum, benign training scenes supplied an empirical tail calibration.

Results:

| Accepted tracks | Raw AUC | Calibrated AUC | Oracle AUC |
|---:|---:|---:|---:|
| 25,000 | 0.593 | 0.560 | 0.666 |
| 50,000 | 0.689 | 0.732 | 0.884 |
| 100,000 | 0.609 | 0.723 | 0.964 |
| 250,000 | 0.592 | 0.720 | 0.967 |

The spatial pathology improved:

```text
calibrated edge@argmax:
0.210 -> 0.170 -> 0.110 -> 0.110 m
```

but the calibrated detector still recovered only part of the available local signal.

Empirical-tail saturation also remained high in roughly one-third of scenes.

---

## 11. Final interpretation

The completed branch supports the following statement:

> **In the current surrogate model, high-Z material produces strong local scattering information, but conventional PoCA-based blind scene search loses a large fraction of that information because spatial localization, coverage, and extreme-value behavior dominate the final scene decision.**

This is not evidence that cosmic-ray muon tomography is ineffective.

It is evidence that the current PoCA scene detector is the wrong endpoint for the research question.

---

## 12. Stop decision

The experiment included an explicit stop rule.

After one diagnosis-derived spatial calibration correction, the branch would stop if:

- scene-level detection remained only moderate,
- the oracle/global gap remained large,
- or another chain of fitted search corrections was required.

Those conditions were met.

The PoCA branch is therefore frozen rather than further tuned.

---

## 13. Next research branch

Any continuation should change the reconstruction/localization method rather than add another threshold to PoCA.

Potential next directions:

- MLEM-style reconstruction
- Bayesian spatial inference
- neural implicit fields
- learned spatial uncertainty maps
- active multi-view acquisition
- measurement selection based on expected information gain

A higher-fidelity validation step using Geant4 or equivalent particle transport is required before physical performance claims.

The most interesting future question produced by this work is:

> **If the local scattering information exists but some regions are poorly localized, can the acquisition geometry be chosen adaptively to reduce spatial uncertainty?**
