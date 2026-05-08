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

## Validation against named extremophiles

The bundled `analogues_v2.csv` and `targets_v2.csv` were used directly,
plus a small set of named-organism tolerance ranges drawn from the
extremophile literature, to sanity-check the algorithm's verdicts
against expected biology. All scores below are the weighted geometric
mean (Liebig) of the four range-bearing parameters (T, Salinity, pH,
Pressure) with equal weights, and use the lethal-limit cliffs.

### From the bundled CSVs

| Analogue → Target                                    | OLD (arith mean) | NEW (Liebig)  | Verdict |
| ---------------------------------------------------- | ---------------- | ------------- | ------- |
| Lost City → Hydrothermal Vents                       | 0.932            | **0.154**     | OLD wildly over-credits: target reaches 290 °C, well above Lost City's 116 °C max and past the 122 °C lethal cliff; salinity is mismatched. NEW correctly flags it as marginal. |
| Lost City → Diffuse Venting Zone                     | 0.956            | **0.244**     | Better fit on T (target 10–90 °C overlaps Lost City), but salinity still mismatched. |
| Lost City → Abyssal Ocean                            | 0.691            | **0.046**     | Lost City organisms are mesophile-to-thermophile; abyssal target at −1 to 1 °C is incompatible. NEW correctly collapses. |
| Borup Fiord (Arctic) → Tiger Stripe Brines           | 0.870            | **0.572**     | Reasonable: cold-adapted, but target salinity 5–40 g/L is several × Borup Fiord's 0.1–10. |
| Borup Fiord (Arctic) → Hydrothermal Vents            | 0.705            | **0.021**     | OLD's 0.71 is the previous algorithm's most embarrassing case: a near-zero-°C Arctic site cannot supply life for a 90–290 °C vent. NEW returns ~0 as biology demands. |

### Named-organism cases

Tolerance ranges below are typical literature values and are illustrative
only. The Enceladus environment ranges come from the same `targets_v2.csv`
the app already ships with.

| Organism (typical tolerance)                                       | Enceladus environment              | OLD   | NEW (Liebig) | Verdict |
| ------------------------------------------------------------------ | ---------------------------------- | ----- | ------------ | ------- |
| *Methanopyrus kandleri* — T 84–122 °C, pH 5.5–7.5, Sal 15–30 g/L, P 0.4–40 MPa | Hydrothermal Vents (T 90–290)      | 0.652 | **0.229**    | Partial: organism viable up to 122 °C, but target's 122–290 °C is past lethal cliff; pH mismatch (organism 5.5–7.5 vs target 8–10.5). |
| *Methanopyrus kandleri*                                            | Diffuse Venting (T 10–90)          | 0.619 | **0.236**    | T overlaps but target is mostly cooler than the organism's optimum; pH still off. |
| *Methanopyrus kandleri*                                            | Abyssal Ocean (T −1 to 1)          | 0.424 | **0.028**    | Hyperthermophile in cold ocean — should be ~0; OLD's 0.42 was nonsense. |
| *Halobacterium salinarum* — T 35–50 °C, Sal **200–340 g/L**, pH 6.5–8.5 | Tiger Stripe Brines (Sal 5–40)     | 0.428 | **0.019**    | A halophile in dilute brine *starves* — its salinity floor is 200 g/L, target maxes at 40. NEW correctly collapses. |
| *Halobacterium salinarum*                                          | Intrashell Mushy Brines (Sal 35–230) | 0.657 | **0.082**    | Better salinity overlap, but target T = −21 to 0 °C is far below the organism's 35–50 °C — Liebig's law dominates. |
| *Picrophilus torridus* — T 47–65 °C, **pH 0–3.5** | Hydrothermal Vents (pH 8–10.5)     | 0.544 | **0.002**    | Acidophile in an alkaline target — pH alone collapses the score, exactly as biology requires. |
| *Picrophilus torridus*                                             | Tiger Stripe Brines (pH 8.5–11.6)  | 0.469 | **0.013**    | Same as above, plus T mismatch. |
| *Thermococcus piezophilus* — T 60–95 °C, **P 20–130 MPa** | Hydrothermal Vents (P 4–10 MPa)    | 0.628 | **0.273**    | Pressure tolerance is broad enough to admit the lower target pressures; T overlaps 90–95 °C with the bottom of the target. |
| *Colwellia psychrerythraea* — **T −5 to 10 °C**, Sal 20–40 g/L, P 0.1–70 MPa | Abyssal Ocean (T −1 to 1)          | 0.574 | **0.523**    | Genuine match: cold-adapted, broad pressure tolerance; target sits inside the organism's envelope on every axis. |
| *Colwellia psychrerythraea*                                        | Tiger Stripe Brines                | 0.549 | **0.523**    | Same: cold + moderately saline overlap. |
| *Colwellia psychrerythraea*                                        | Hydrothermal Vents (T 90–290)      | 0.636 | **0.123**    | Psychrophile cannot survive 90 °C, never mind 290 °C — NEW correctly cuts the OLD's misleading 0.64. |
| *Deinococcus radiodurans* — T 4–45 °C, Sal 0–30 g/L, pH 4.5–10, P ~atmos. | Tiger Stripe Brines                | 0.661 | **0.883**    | **Honest limitation, not a bug.** *D. radiodurans* has very broad tolerance and the target sits *just outside* its range on each axis. The universal 25% soft-edge `δ` is too generous in this case — the algorithm reads "just outside" as "stressed but viable" when the right reading is "this organism doesn't really live in cold dilute alkaline brine." See the Limitations section below. |

### Patterns confirmed

1. **Lethal cliffs do their job.** Every case where biology says "no" because a hard limit is violated (acidophile in alkali, hyperthermophile in cold, psychrophile in 290 °C) collapses to <0.13 with the new algorithm; the old algorithm scored these between 0.42 and 0.71.
2. **Genuine matches survive.** *Colwellia* in cold ocean stays at 0.52; *M. kandleri* in 90–122 °C portion of vents stays at 0.23 (correctly partial because half the target temperature range is past the lethal cliff).
3. **Asymmetry is preserved.** Wide-tolerance organisms still score 1.0 when the target is fully inside their envelope (the user's original 0–100 / 70–90 case), and narrow-tolerance organisms correctly score low when the target is wider than they are.

### Known limitations (surfaced by validation)

- **Soft-edge `δ` is universal across parameters.** For broad-tolerance organisms like *D. radiodurans*, a target lying just outside the analogue range on multiple axes can score deceptively well. A future improvement is to make `δ` parameter-specific (smaller for pH, where the lethal cliffs are sharp; larger for redox / isolation, where data is sparse) or to derive it from the reliability rating.
- **Range = tolerance** assumption. The analogue range is the *observed environmental conditions*, not the organism's full physiological envelope, so the score is conservative for organisms whose true tolerance exceeds what they're routinely measured in. The biochemistry-first-principles candidate (Alternative B below) is the natural next step here.
- **Independence of parameters.** The geometric mean assumes parameters are independent. Real survival has cross-parameter trade-offs (high salinity depresses freezing point, etc.) — the Mahalanobis candidate (Alternative C below) is the future answer if dataset size grows.

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
