# AstroMatch Scoring Theory & Improvements

## Why the original `calculate_suitability` was too primitive

The previous function (Gaussian-of-midpoints OR Jaccard-of-ranges, whichever
is larger) had two independent flaws:

1. **It was symmetric.** Survival is not a similarity question. If an
   analogue organism tolerates 0–100 °C and the target environment is
   70–90 °C, a biologist would say the score is 100% — the target
   conditions sit fully inside the organism's tolerance envelope. The old
   function returned **0.325** for that case, because the midpoints (50 vs
   80) are 30 °C apart and Jaccard's union punishes the wide analogue.

2. **It aggregated by weighted arithmetic mean.** A 100% fit on five
   parameters cannot rescue a lethal sixth: a halophile dropped into 200 °C
   water dies regardless of how perfectly its pH, salinity, pressure,
   redox, and isolation match. The old aggregator at line 162 quietly
   granted compensability that biology forbids.

The user's stated goal — "determine compatibility of a lifeform on Earth
surviving in that scenario" — is therefore a two-layer problem, and the
right fix touches both layers.

## The two range types

The fix also takes seriously the asymmetry of *what the ranges mean*:

- **Earth analogue ranges** are real, measured tolerance windows. The
  organisms living there demonstrably survive across that span.
- **Target (extraterrestrial) ranges** are partly tolerance information
  and partly *epistemic uncertainty* about conditions we cannot directly
  sample.

The new model treats the analogue range as a tolerance envelope and asks
how much of the target range falls inside that envelope. A target
distribution that is *narrower* than the analogue and *contained* by it is
a maximal match (1.0); a target that extends outside the envelope is
penalised in proportion to how far outside it strays.

## The chosen approach

A fusion of three of the six candidates studied:

### 1. Asymmetric Tolerance Envelope (per-parameter primitive)
For each parameter we treat `[site_min, site_max]` as the inferred
tolerance envelope. The membership function is

```
μ(x) = 1                                 if site_min ≤ x ≤ site_max
μ(x) = exp(-((site_min − x) / δ)²)       if x < site_min
μ(x) = exp(-((x − site_max) / δ)²)       if x > site_max
```

with bandwidth `δ = 0.25 · max(site_max − site_min, 1)` — wide enough to
admit "stressed but viable" conditions a small distance outside the
observed range, narrow enough that a target sitting far outside the
envelope decays to near-zero. The score is the mean of `μ` over the
target interval, computed in closed form (the Gaussian tails are exact
`erf` differences — no numerical integration, no scipy dependency).

This is **asymmetric on purpose**. Swapping (site, target) gives a
different answer because the question is directional: does *this
organism* survive *those conditions*?

### 2. Liebig geometric-mean aggregator
The per-parameter fits are aggregated as a weighted geometric mean —
mathematically the joint survival probability under the (admittedly
strong) assumption of parameter independence:

```
S_total = exp( Σ (wᵢ / Σw) · log(max(fitᵢ, ε)) )
```

This collapses toward zero whenever any single fit does, so a near-lethal
parameter dominates the score the way Liebig's Law of the Minimum says it
should. The ε-floor (1e-3) keeps `log` defined on hard-zero fits while
still propagating their dominance.

### 3. Lethal-limit short-circuit
A small dictionary of universal-life lethal limits is consulted before
scoring. Targets entirely outside life's known envelope short-circuit to
zero; targets that straddle the cliff have their score reduced
proportionally to the viable fraction. Sources for each constant are in
`LETHAL_LIMITS` in `astromatch_v2.py`.

| Parameter   | Lower lethal | Upper lethal | Source                                                                       |
|-------------|--------------|--------------|------------------------------------------------------------------------------|
| Temperature | −25 °C       | 122 °C       | Cold: Toner et al. 2014 (Mars perchlorate brines). Hot: Takai et al. 2008 (*M. kandleri* strain 116 at 122 °C / 20 MPa, PNAS 105:10949). |
| pH          | 0.0          | 12.5         | Acid: Schleper et al. 1995 (*Picrophilus torridus*, J. Bacteriol. 177:7050). Alkali: Suzuki et al. 2021 (*Serpentinimonas maccroryi* B1, pH 12.5 record); genus described in Suzuki et al. 2014, *Nat. Commun.* 5:3900. |
| Pressure    | 0 MPa        | 1100 MPa     | Sharma et al. 2002, *Science* 295:1514. Active formate oxidation observed up to ~1060 MPa; 1100 MPa is the conservative active-growth cap. |
| Salinity    | —            | —            | No universal upper cap; halophiles grow to NaCl saturation.                  |
| Isolation   | —            | —            | Ordinal score; no biological universal.                                      |
| Redox       | —            | —            | Ordinal score; no biological universal.                                      |

> **A note on these constants.** They were checked against published
> sources (May 2026) and are conservative, but specific numeric edges
> are inherently judgement calls — the *Picrophilus* paper reports growth
> down to pH 0.03 with adapted strains, the Sharma paper reports cell
> *viability* (not active metabolism) up to ~1600 MPa, etc. The intent
> is a defensible default, not a published constants table. A domain
> expert should review these before any scientific use.

## Worked example

Organism 0–100 °C survives in biome 70–90 °C:

- Old score: **0.325** (biologically wrong)
- New score: **1.000**

Organism 30–50 °C facing biome 0–100 °C (the asymmetric reverse):

- Old score: **0.995** (over-claims survival — a narrow specialist
  cannot survive a much wider environmental range)
- New score: **0.289** (correctly penalises the parts of 0–30 and 50–100
  that the organism never experienced)

## Alternatives considered (the five strong-but-not-chosen candidates)

Six modelling approaches were investigated by independent research
agents. Each is defensible in its own right, and any of them would be a
substantial improvement over Gaussian-Jaccard. They are documented here
because the chosen fusion borrows ideas from several, and because future
versions of AstroMatch may want to layer one on top of the current
model.

### A. Probabilistic / Bayesian (uniform / triangular / truncated-normal)
Model both site and target as PDFs and compute `P(target ∈ site
tolerance)` in closed form. The uniform-uniform special case reduces
exactly to directed-overlap fraction — i.e., a special case of the
chosen tolerance envelope. The richer triangular and truncated-normal
variants would add a "stress at extremes" prior (lower membership near
range boundaries). Rejected as primary because at uniform priors it is
mathematically equivalent to what we already do, and richer priors are
hard to defend with N=14. Strong candidate for a future "advanced
priors" toggle.

### B. Physiological / Biochemical First Principles
The most defensible approach to an astrobiology reviewer: per-parameter
survival functions derived from biochemistry (Q10/Arrhenius for
temperature with a hard denaturation cliff at 122 °C, Raoult-derived
water activity for salinity with an `a_w ≈ 0.605` floor, log-distance
for pH with cytoplasmic homeostasis cost, Gibbs-energy availability for
redox, Yayanos / Dalmasso piezophile thresholds for pressure). The
strongest scientific story of all six. Rejected as primary because it
encodes a specific Earth-water-carbon biology, requires per-parameter
calibration, and adds 80–100 LOC plus a constants module. Implementing
it as a `--biochemistry-strict` mode is a strong follow-up.

### C. Mahalanobis / Multivariate Covariance-Aware Distance
Treats parameters as correlated (salinity-temperature freezing-point
coupling, pH-redox Pourbaix coupling, pressure-temperature water-phase
coupling) and scores via a single Mahalanobis distance. Rejected because
N=14 is far too small to estimate a 6×6 covariance robustly — even with
shrinkage toward an astrobiology prior — and because the per-parameter
UI (radar chart, parameter breakdown) becomes a mathematical fiction
once axes are entangled. The hybrid 3-axis variant (T, log-salinity,
log-pressure only) is interesting and could be added as an optional
"physical-state coupling" diagnostic.

### D. Mahalanobis with shrinkage prior
A subset of (C) — same conclusion. Documented separately because the
shrinkage estimator is a standard workhorse and would be the natural
next step if N grows.

### E. Trapezoidal Fuzzy Membership with literature lethal limits
Functionally close to the chosen envelope, but with linear ramps to a
hand-coded outer-lethal boundary instead of Gaussian decay. More
explainable to non-experts ("inside = 1, outside = lethal limit, ramp
between"), but symmetric and so does not address the directional bug
that motivated this rewrite. The lethal-limit table from this proposal
is the source of the `LETHAL_LIMITS` dictionary used in the chosen
implementation.

### F. Pure Liebig Min Aggregation
Pure `min(fits)` instead of geometric mean. Maximally faithful to
Liebig's Law but discontinuous and information-discarding. The chosen
geometric mean is a smoothed Liebig that approaches `min` behaviour as
fits decline while still differentiating between two sites that differ
only in their non-limiting parameters.

## Implementation notes

- Function signature is preserved; `lethal_min`/`lethal_max` are optional
  kwargs so the call site is the only change required.
- No new dependencies (`math.erf` is stdlib).
- NaN-passthrough behaviour is preserved (`None` indicates missing data,
  flagged downstream).
- The score remains in `[0, 1]`, drop-in compatible with the existing
  radar chart, parameter-breakdown table, and "X% match" verdict text.
- Reliability scores (`*_rel`) are not yet folded into `δ`; this is a
  natural next improvement (lower reliability → wider `δ` → more
  forgiving edges) and was identified by the Bayesian-candidate agent.
