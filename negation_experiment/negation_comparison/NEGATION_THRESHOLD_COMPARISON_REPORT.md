# Scope vs NLI Negation — Threshold & Multi-Paper Report

## The two detectors

| | **NLI negation** | **Scope negation / speculation** |
|---|---|---|
| Model | `cross-encoder/nli-MiniLM2-L6-H768` | Fine-tuned PubMedBERT token classifier |
| Unit of analysis | Entity clause (after contrastive split) | Full sentence token scopes |
| Decision | Contradiction score vs hypothesis > threshold → `negated` | Entity overlap with negation / speculation scope blocks |
| Outputs | `negated`, `confidence` | `scope_negated`, `scope_speculative`, overlaps |
| Controlled by | `NEGATION_THRESHOLD` | Scope overlap thresholds |

---

## Section 1 — Threshold ablation (original study, 3 papers)

**Variable changed:** `NEGATION_THRESHOLD` only (`0.45` → `0.60`). Scope post-processing held constant.

### Overall results (1,407 entities)

| Metric | NLI @ 0.45 | NLI @ 0.60 | Change |
|---|---:|---:|---|
| NLI–scope agreement | 1,355 / 1,407 (**96.3%**) | 1,375 / 1,407 (**97.7%**) | **+1.4 pp** |
| NLI negated entities | 45 | 25 | **−20 (−44%)** |
| Scope negated entities | 13 | 13 | **0** (unchanged) |
| Scope speculative entities | 88 | 88 | **0** (unchanged) |
| Disagreements | 52 | 32 | **−20 (−38%)** |

**Summary:** Raising the threshold made NLI stricter. Scope stayed fixed. Agreement rose because NLI stopped calling borderline cases (0.46–0.58) negated.

### What changing the NLI threshold does

```
negated = (contradiction_score > NEGATION_THRESHOLD)
```

| Threshold | Effect on NLI |
|---|---|
| **0.45** | Higher recall — marks weak/borderline contradictions (~0.46–0.58) often speculation or background |
| **0.60** | Higher precision — only strong contradictions count; borderline scores stay `PRESENT` |

### Confidence bands filtered at 0.60

| NLI score band | Count removed | Typical content |
|---|---:|---|
| 0.45 – 0.49 | 5 | Background / weak uncertainty |
| 0.50 – 0.54 | 10 | Speculative or intro clauses |
| 0.55 – 0.59 | 5 | Borderline review language |

---

## Section 2 — Expanded evaluation (5 papers, 2,355 entities)

Includes the two papers added (PMC13083324, PMC11097689), all run at `NEGATION_THRESHOLD=0.60` with the updated scope post-processing (`_MAX_LEN=512`).

### Per-paper results @ 0.60

| Paper | Entities | NLI neg | Scope neg | Scope spec | Agreement | Disagree |
|---|---:|---:|---:|---:|---:|---:|
| PMC8160999 | 102 | 1 | 4 | 9 | 97 / 102 (95.1%) | 5 |
| PMC11975295 | 260 | 13 | 1 | 30 | 246 / 260 (94.6%) | 14 |
| PMC12659517 | 1,045 | 11 | 9 | 49 | 1,031 / 1,045 (98.7%) | 14 |
| PMC13083324 | 747 | **60** | 4 | 78 | 689 / 747 (92.2%) | **58** |
| PMC11097689 | 201 | 15 | 10 | 26 | 188 / 201 (93.5%) | 13 |
| **TOTAL** | **2,355** | **100** | **28** | **192** | **2,251 / 2,355 (95.6%)** | **104** |

### Key observations from new papers

**PMC13083324** has the most NLI negations (60) and most disagreements (58) of any paper tested. This is driven by two distinct phenomena — a new false-positive pattern and legitimate high-confidence detections — described in the failure taxonomy below.

**PMC11097689** shows scope's strongest performance: 10 scope-negated entities, with 6 cases where both detectors agree and 4 genuine scope-only catches NLI misses.

---

## Section 3 — Failure pattern taxonomy (updated across all 5 papers)

### Pattern A — Distributed predicate-level negation (NLI catches, Scope misses)

Negation expressed through meaning spread across the clause, no local cue adjacent to entity.

| Entity | Paper | NLI conf | Sentence fragment |
|---|---|---:|---|
| `dementia` | PMC11975295 | 0.923 | *"effective treatments have been **elusive**"* |
| `amyloid` | PMC11975295 | 0.981 | *"clinical trials … **yielded disappointing results**"* |
| `CSF` | PMC12659517 | 0.931 | *"full spectrum … **remains poorly understood**"* |
| `vitamin E`, `selegiline` + 3 more | PMC11097689 | 0.994 | *"**no clear evidence** of any beneficial effects"* |
| `Tramiprosate`, `Avagacestat` etc. | PMC13083324 | 0.961–0.963 | *"**Failed** in Phase III"*, *"**lack of efficacy**"* |

**Why scope misses:** The negation cue is a verb/adjective of failure or inadequacy, not a standard scope-trigger word (`not`, `without`, `no`). The entity is not inside a predicted negation-scope block.

---

### Pattern B — Local explicit-cue negation (Scope catches, NLI miss)

Negation expressed through a standard cue word (`without`, `no`, `not associated with`, `neither…nor`) that places the entity inside the scope block.

| Entity | Paper | Scope overlap | NLI conf | Sentence fragment |
|---|---|---:|---:|---|
| `dasatinib`, `quercetin`, `INK-ATTAC` | PMC12659517 | 1.0 | **0.033** | *"**neither** … **nor** genetic clearance … **ameliorated** EAE severity"* |
| `human healthy breast cells` | PMC8160999 | 0.885 | 0.234 | *"**without** any cytotoxic effects on …"* |
| `disease` | PMC11975295 | 1.0 | 0.038 | *"limited … **without** altering disease progression"* |
| `impaired consciousness` | PMC11097689 | 0.955 | **0.192** | *"**not associated with** impaired consciousness"* |
| `estrogen`, `Alzheimer's disease` | PMC11097689 | 1.0 / 0.947 | 0.981 | *"**There is no evidence that** estrogen … is effective"* ✓ both |
| `ginkgo biloba`, `B vitamins`, `omega-3` | PMC11097689 | 0.923–0.900 | 0.956 | *"**no evidence** from clinical trials that …"* ✓ both |
| `anti-inflammatory` | PMC13083324 | 1.0 | 0.660 | *"**no** anti-inflammatory agent has yet been approved"* ✓ both |

**Note:** When the cue is `"no evidence that/from"` (a strong lexical signal), both detectors agree at high confidence. When the cue is local syntactic (`without`, `not associated with`, `neither…nor`), NLI drops to near-zero while scope fires correctly.

---

### Pattern C — NLI false positives on enumeration / listing sentences (NEW — PMC13083324)

A systematic error specific to review-paper style: sentences that enumerate competing hypotheses or list drug candidates trigger NLI negation on **all entities in the sentence** at the same confidence score.

| Sentence | # Entities wrongly negated | NLI conf |
|---|---:|---:|
| *"Multiple hypotheses, including the amyloid cascade, tau, cholinergic, vascular, and neuroinflammatory theories, are **discussed** to elucidate disease pathogenesis."* | **5** | 0.611 |
| *"Drugs targeting TNF-α, IL-1β, and other inflammatory mediators are **under investigation**…"* | **5** | 0.720 |
| *"Tarenflurbil (Flurizan) **failed** in Phase III clinical trials"* | 4 | 0.762 |
| *"Verubecestat **Failed** in Phase III … worsening cognition and side effects (weight loss, liver toxicity)"* | 4 | 0.716 |

The first two are clear false positives — entities are being *discussed*, not negated. The last two are real drug failures but NLI propagates the negation signal to **side effects** (e.g., `weight loss`, `liver toxicity`) which are not negated — they occurred.

**Why this happens:** The NLI model receives the full clause as a hypothesis about each entity. When a sentence has many entities, the same clause scores above threshold for all of them, creating a spurious cluster. Scope avoids this because it only fires when the entity token sits inside a labelled scope block.

**Impact on PMC13083324:** Of the 60 NLI negations, approximately **6 are clear false positives** (conf 0.61), **19 are uncertain** (conf 0.65–0.79, include side-effect propagation and investigational-drug framing), and **35 are likely correct** (conf ≥ 0.80, confirmed drug failures with explicit language).

---

### Pattern D — NLI/Scope inversion: speculative vs negated

Cases where NLI calls `negated=True` but Scope correctly identifies `speculative=True`, indicating uncertainty rather than absence.

| Entity | Paper | NLI conf | Scope | Sentence fragment |
|---|---|---:|---|---|
| `CNS immune` | PMC12659517 | 0.896 | spec=True (0.9) | *"**determining if** and **how** these hallmarks cause …"* |
| `glia cell` | PMC12659517 | 0.970 | spec=True (0.889) | *"**whether** AP-1 … **remains unknown**"* |
| `patients` | PMC11097689 | 0.839 | spec=True (1.0) | *"novel biomarkers **could have** a role … **require validation**"* |
| `disease` | PMC13083324 | 0.953 | spec=True (1.0) | *"exact mechanisms … **are not fully understood**"* |

In these cases, **Scope is more correct than NLI** — the entity is under uncertainty framing, not outright negation.

---

## Section 4 — Detector complementarity (updated)

Across all 5 papers:

| Scenario | Count | NLI verdict | Scope verdict |
|---|---:|---|---|
| Both agree: negated | ~22 | `negated=True` | `scope_negated=True` |
| NLI only: high-conf negation (≥0.80) | ~60 | `negated=True` | `scope_negated=False` — distributed/failure language |
| NLI only: false positive cluster (0.60–0.65) | ~6+ | `negated=True` | both False — enumeration sentences |
| NLI only: spec/neg inversion | ~4 | `negated=True` | `scope_speculative=True` — should be POSSIBLE |
| Scope only: local cue (NLI < 0.30) | ~8 | `negated=False` | `scope_negated=True` (overlap ≥ 0.85) — `without/neither…nor/not associated with` |
| Both agree: PRESENT | majority | `negated=False` | both False |

**Scope is the higher-precision detector.** When scope fires at high overlap (≥ 0.85) with NLI confidence < 0.30, it is almost always correct (see Pattern B). When NLI fires alone, correctness depends heavily on confidence and sentence type.

---

## Section 5 — NLI false positive rate estimate (updated)

| Paper type | NLI FP estimate | Driver |
|---|---|---|
| Targeted experiment / methods papers (PMC8160999, PMC12659517) | Low (~1–2 FP) | Fewer enumerations, more direct negation language |
| Review papers (PMC11975295, PMC11097689) | Moderate (~3–5 FP) | Speculative framing, hypothesis listings |
| Comprehensive review with drug trial tables (PMC13083324) | **High (~25 FP)** | Enumeration clusters + side-effect propagation + "discussed/investigated" sentences |

This suggests NLI reliability degrades on **review-style papers** that list competing theories and failed clinical trials, where the contrastive clause contains many entities simultaneously.

---

## Section 6 — Conclusions

1. **NLI @ 0.60 is still the stronger primary negation detector** across all 5 papers (100 negations vs 28 for scope), with good precision on explicit failure/absence language.
2. **NLI has a new identified failure mode** on review papers: enumeration sentences cause multi-entity false positive clusters at conf ~0.61–0.72.
3. **Scope remains essential for local-cue negation** (`without`, `neither…nor`, `not associated with`). NLI scores for these patterns are near zero (0.003–0.234) — it is blind to them.
4. **Scope is more reliable on speculative inversion** — where NLI miscalls `negated=True` but the correct label is `POSSIBLE`. Scope's `speculative=True` flag is accurate for `whether/if/determining if` framing.
5. **The `"no evidence that/from"` pattern is the strongest agreement zone** — both detectors agree with high confidence because it combines an explicit `no` cue (scope trigger) + strong NLI contradiction score.
6. **Remaining scope recall gap is a training-data problem**, not a post-processing problem. Post-processing bugs fixed  produced no measurable recall change because the bottleneck is model scope predictions, not character mapping.

---

## Files

```
negation_experiment/negation_comparison/
├── PMC8160999.md
├── PMC8160999.json
├── PMC11975295.{json,md}
├── PMC12659517.{json,md}
├── PMC13083324.{json,md}   
├── PMC11097689.{json,md}  
└── NEGATION_THRESHOLD_COMPARISON_REPORT.md
```

---

## Reproduce

```bash
# Original threshold ablation
NEGATION_THRESHOLD=0.45 SCOPE_NEGATION_ENABLED=true \
  .venv/bin/python negation_experiment/scripts/compare_negation_pmc.py PMC8160999

NEGATION_THRESHOLD=0.60 SCOPE_NEGATION_ENABLED=true \
  .venv/bin/python negation_experiment/scripts/compare_negation_pmc.py PMC8160999

```
