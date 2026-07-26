# NER Comparison Results

This folder contains side-by-side entity recognition comparisons for two models:

- `scispacy`
- `hunflair`

The comparisons were run with the `ensemble_consensus_2plus` evaluation method.
This README reflects the latest results currently stored in this folder.

## What we used for comparison

The results in this folder are interpreted using four comparison lenses:

1. Net Entity Yield
   - What it measures: how many new unique entities one model adds compared with the other.
   - Formula: `Gain(Model A) = |Entities_ModelA \ Entities_ModelB|`
   - Why it matters: shows how much extra coverage a model contributes without assuming gold labels.

2. Category-Specific Coverage Shift
   - What it measures: which entity types each model finds more often, such as `DISEASE`, `CHEMICAL`, `GENE_OR_GENE_PRODUCT`, and `CELL_LINE`.
   - Formula: `Coverage(T, M) = Count of entities of type T extracted by model M`
   - Why it matters: shows whether one model fills blind spots left by the other.

3. High-Confidence Consensus Precision
   - What it measures: agreement-based precision using the shared consensus set as the strict proxy for true positives.
   - Formula: `TP proxy = Entities_scispacy ∩ Entities_HunFlair`
  - Why it matters: avoids treating every unique entity from one model as a false positive when the other model simply missed it.
  - Note: the current metrics files use the ensemble consensus proxy directly; if a confidence score is available in future runs, it can be folded into this lens as an additional filter.

4. Span Precision and Boundary Quality
   - What it measures: whether the model extracts complete entity spans or fragmented partial spans.
   - Formula: `Avg Length(M) = (Σ length(e)) / |M|`
   - Why it matters: longer, cleaner spans are usually more useful for downstream relation extraction and LLM-based analysis.

## What is here

Each paper has:

- a raw JSON result file: `PMC*_*.json`
- a metrics file: `PMC*_*_metrics.json`

The metrics files are the easiest way to read the outcome of each comparison.

## Papers included

The current comparison set covers 6 PMC articles:

- `PMC8160999`
- `PMC9500000`
- `PMC11097689`
- `PMC11975295`
- `PMC12659517`
- `PMC13083324`

## Overall summary

Across all 6 papers in the latest run:

- Total unique entities found by the ensemble: `2,568`
- Total consensus entities: `322`
- Average unique entities per paper: `428.0`
- Average consensus entities per paper: `53.7`

Model-level totals across the 6 papers:

- `scispacy`
  - Total predicted entities: `2,405`
  - Exclusive entities: `2,083`
  - Average span length: `12.5` to `15.1` tokens depending on paper
  - Average overlap ratio: `0.215`
- `hunflair`
  - Total predicted entities: `485`
  - Exclusive entities: `163`
  - Average span length: `9.0` to `15.5` tokens depending on paper
  - Average overlap ratio: `0.652`

## Main takeaway

The two models behave very differently:

- `scispacy` is much more expansive and produces far more entity spans.
- `hunflair` is more selective and usually yields higher-precision biological/medical entities.
- In these comparisons, `hunflair` generally shows stronger `DISEASE` performance.
- `scispacy` is often stronger on broad coverage, especially for long or noisy spans, citations, and context-heavy mentions.

## Precision and Recall Interpretation

Based on the comparison results in this folder:

- `hunflair` is usually the better choice when you want **higher precision** for clinical and biomedical entity types, especially `DISEASE`, `CHEMICAL`, and `ORGANISM`.
- `scispacy` often has **lower precision** because it predicts many more spans, including noisy or overly broad matches.
- `scispacy` can still be useful when you want **higher coverage** and are willing to accept more false positives.
- `GENE_OR_GENE_PRODUCT` is the weakest area for both models overall, with `scispacy` sometimes having the edge on coverage but not consistently on precision.

In simple terms:

- **Higher precision:** `hunflair` on most biomedical labels in these results
- **Broader coverage, more noise:** `scispacy` on many of the same labels, especially where it over-generates spans
- **Mixed behavior:** `GENE_OR_GENE_PRODUCT`, where neither model is consistently strong across papers

## Average F1 by label

Average F1 across the available papers for each label:

| Label | scispacy | hunflair |
| --- | ---: | ---: |
| `CELL_LINE` | `0.286` | `0.400` |
| `CHEMICAL` | `0.562` | `0.637` |
| `DISEASE` | `0.467` | `0.758` |
| `GENE_OR_GENE_PRODUCT` | `0.310` | `0.220` |
| `ORGANISM` | `0.453` | `0.752` |

## Per-paper highlights

### `PMC8160999`

- Strongest `scispacy` result on `ORGANISM` with F1 `0.923`
- `hunflair` performed best on `CHEMICAL` and `ORGANISM` in this paper
- This paper shows one of the clearest cases where `hunflair` is more precise for biomedical terms

### `PMC9500000`

- `scispacy` found many more entities, but with a low overlap ratio (`0.125`)
- `hunflair` was notably better on `DISEASE` and `ORGANISM`
- Both models struggled on `GENE_OR_GENE_PRODUCT`

### `PMC11097689`

- `hunflair` beat `scispacy` on `CHEMICAL` and `DISEASE`
- `scispacy` was slightly better on `ORGANISM`
- This paper has fairly balanced results between the two models

### `PMC11975295`

- `hunflair` was stronger on `DISEASE`
- `scispacy` was stronger on `GENE_OR_GENE_PRODUCT`
- Both models found the same number of `DISEASE` consensus entities, but `hunflair` had better precision

### `PMC12659517`

- Largest entity volume in the set
- `scispacy` produced extremely broad coverage, with `1,098` total entities
- `hunflair` was much more selective and achieved perfect `ORGANISM` F1 in this paper

### `PMC13083324`

- `hunflair` clearly outperformed `scispacy` on `CHEMICAL`, `DISEASE`, and `ORGANISM`
- `scispacy` remained better on `GENE_OR_GENE_PRODUCT`
- This is one of the strongest papers for `hunflair` overall

## Notes

- All reported precision, recall, and F1 values in the metrics files use the `2plus` consensus setup.
- The `exclusive_gains` arrays in each metrics file list example spans found by only one model.
- The raw JSON files can be used if you need to inspect the actual entity-level predictions.
- Recall here is a consensus-proxy score, not gold-standard recall.
