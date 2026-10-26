# Layer 4 Baseline Evaluation Report (Report A)

**Date**: 2026-10-25  
**Working Directory**: `/home/muaz/projects/Icog/bio-semantic-parser`  
**Current Branch**: `layer4-ner-experiments`  
**Base Commit**: `ed899eb83812aa3e5e262be3a4c2cc992abeeb1e` (`Merge pull request #32 from Abdu1964/main`)  
**Target Upstream Branch for Final Contribution**: `origin/experiment`  

---

## 1. Environment & Hardware Context

- **Operating System**: Linux (Ubuntu x86_64, Kernel 6.8.0-45-generic)
- **CPU**: 8 vCPUs (AMD/Intel x86_64)
- **RAM**: 14 GiB total (9.6 GiB available, 5.3 GiB used)
- **Storage**: 468 GB NVMe (324 GB available)
- **GPU**: No GPU detected / CUDA unavailable (`torch.cuda.is_available()` returns `False`)
- **Python Environment**:
  - Python version: 3.10.22 (in `.venv`), System Python 3.14.4
  - Active environment: Local virtual environment at `.venv/`
  - Current dependency state: Virtual environment is partially provisioned. Core packages such as `pydantic (2.14.1)`, `pytest (9.1.1)`, `PyYAML (6.0.3)`, `requests (2.34.2)`, and `rich (15.0.0)` are present.
  - **Critical Dependency Deficiency**: Heavy machine learning and biomedical NLP packages declared in `requirements.txt` are missing:
    - `spacy` (pinned to `3.7.5` in requirements) is **not installed**.
    - `scispacy` and the 4 scispaCy models (`en_ner_bc5cdr_md`, `en_ner_jnlpba_md`, `en_ner_bionlp13cg_md`, `en_core_sci_lg`) are **not installed**.
    - `gliner` is **not installed**.
    - `transformers` and `torch` are **not installed** in the virtual environment.
  - **Local Model Cache State**:
    - Directory `~/.cache/huggingface/hub/` contains `sentence-transformers--all-MiniLM-L6-v2` only.
    - Required Layer 4 models `cross-encoder/nli-MiniLM2-L6-H768`, `d4data/biomedical-ner-all`, and `Ihor/gliner-biomed-base-v1.0` are **not cached locally**.

---

## 2. Description of the Current NER Implementation

Layer 4 pre-extraction coordinates entity recognition across four modular components governed by [`src/preextraction/preextractor.py`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/preextractor.py):

### 2.1 Multi-Model scispaCy Ensemble
- **Implementation**: [`Preextractor._run_ensemble`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/preextractor.py#L49-L71) dynamically scans environment variables `NER_MODEL_1` through `NER_MODEL_20`, with default slots:
  - `NER_MODEL_1`: `en_ner_bc5cdr_md` (Chemicals, Diseases)
  - `NER_MODEL_2`: `en_ner_jnlpba_md` (DNA, RNA, Cell Lines, Cell Types, Proteins)
  - `NER_MODEL_3`: `en_ner_bionlp13cg_md` (Cancer Genetics, Cellular Components, Species)
  - `NER_MODEL_4`: `en_core_sci_lg` (General Biomedical spans)
- **Span Aggregation**: Spans from each model are aligned on a base spaCy document using `char_span(start, end, alignment_mode="expand")` and greedily deduplicated using `spacy.util.filter_spans()`.
- **Dict Conversion & Filter**: [`NERTagger.from_doc(doc)`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/ner_tagger.py#L21-L42) converts spans into standard dictionaries:
  - Rejects text with stripped length `< 2`, starting with `[` or `(`, or with `< 30%` alphabetic characters.
  - Hardcodes `confidence: 1.0`, `negated: False`, `assertion: "PRESENT"`, `source: "ensemble"`.
  - **Deduplication Flaw**: Deduplicates by string surface text alone (`seen.add(normalized)`), dropping subsequent mentions of identical terms even at distinct token offsets.

### 2.2 HuggingFace Clinical NER
- **Implementation**: [`src/preextraction/hf_ner_tagger.py`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/hf_ner_tagger.py).
- **Default Model**: `d4data/biomedical-ner-all` (configurable via `HF_NER_MODEL`).
- **Target Types**: Supplements clinical categories missing in scispaCy: `Diagnostic_procedure` (`PROCEDURE`), `Therapeutic_procedure` (`CLINICAL_INTERVENTION`), `Medication` (`DRUG`), `Sign_symptom` (`SYMPTOM`), `Disease_disorder` (`DISEASE`), and `Lab_value` (`CLINICAL_MEASUREMENT`).
- **Configuration**: Enabled via `HF_NER_ENABLED` (default `"true"`), score threshold `HF_NER_THRESHOLD` (default `0.80`), capped at `text[:8000]`.
- **Enforcement**: Configured with `TRANSFORMERS_OFFLINE="1"` and `local_files_only=True`. Fails silently (`except Exception: return []`) if model weights are not pre-cached.

### 2.3 GLiNER-BioMed Zero-Shot Tagger
- **Implementation**: [`src/preextraction/gliner_tagger.py`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/gliner_tagger.py).
- **Default Model**: `Ihor/gliner-biomed-base-v1.0` (configurable via `GLINER_MODEL`).
- **Target Types**: Evaluates 36 natural language prompt definitions mapped to upper-case taxonomy labels (`GENE`, `PROTEIN`, `TRANSCRIPT`, `EXON`, `NON_CODING_RNA`, `GENOMIC_VARIANT`, `SEQUENCE_VARIANT`, `REGULATORY_REGION`, `ENHANCER`, `SUPER_ENHANCER`, `PROMOTER`, `TAD`, `EPIGENOMIC_FEATURE`, `MOTIF`, `PATHWAY`, `TISSUE`, `CELL_TYPE`, etc.).
- **Configuration**: Enabled via `GLINER_ENABLED` (default `"true"`), threshold `GLINER_THRESHOLD` (default `0.5`), minimum word length `3`.

### 2.4 PubTator3 Curation Client
- **Implementation**: [`src/preextraction/pubtator_client.py`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/pubtator_client.py).
- **Execution**: Triggered only when `source_name == "pubmed"` and `doc_id` is a numeric string $\le 10$ characters (PMID).
- **Relocalization**: PubTator offsets are document-level. [`_relocalize_to_chunk()`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/preextractor.py#L148-L162) finds the first occurrence via `lower_chunk.find(text.lower())`.

### 2.5 Cross-Source Overlap Resolution
- **Implementation**: [`_resolve_overlaps(entities)`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/preextractor.py#L170-L211).
- **Sorting Rule**: Sorts descending by `(end - start, confidence, _SOURCE_PRIORITY, -start)` where `_SOURCE_PRIORITY` is `{"ensemble": 3, "pubtator3": 2, "gliner": 1, "hf_ner": 0}`.
- **Resolution Strategy**:
  - Exact matches coalesce into winner; conflicting labels saved to `alt_label` and `alt_confidence`.
  - Partial overlaps with higher-priority / longer spans are dropped.

---

## 3. Description of the Current Negation Detector

- **Implementation File**: [`src/preextraction/negation_detector.py`](file:///home/muaz/projects/Icog/bio-semantic-parser/src/preextraction/negation_detector.py).
- **Model**: `cross-encoder/nli-MiniLM2-L6-H768` (Natural Language Inference sequence classification cross-encoder).
- **Inference Mode**:
  - Offline forced (`TRANSFORMERS_OFFLINE="1"`, `local_files_only=True`).
  - Evaluates clause pairs against fixed hypothesis:
    ```
    "This study demonstrates a confirmed biological effect, finding, or association."
    ```
- **Clause Extraction**:
  - Looks up sentence in `doc.sents` based on entity character span.
  - Splits sentence at contrastive conjunctions using regex `_CONTRASTIVE`:
    `r'\b(but not|however|although|whereas|while yet|despite|in contrast|on the other hand|except|rather than|but(?!\s+not)|yet(?!\s+not))\b'`.
  - Extracts the isolated sub-clause containing the entity mention.
- **Classification & Threshold**:
  - Model outputs 3 class logits: `0: contradiction`, `1: entailment`, `2: neutral`.
  - Takes softmax: `contradiction_score = float(probs[0])`.
  - If `contradiction_score > NEGATION_THRESHOLD` (default `0.45`), entity is flagged `negated: True`, `assertion: "ABSENT"`.
- **Downstream Impact**:
  - Layer 6 (LLM Extractor): Injects `[NEGATED]` tag into prompt and adds a chunk-level negation banner.
  - Layer 7 (Confidence Scorer): Caps relation confidence at `0.65` for any relation where `negated == True`.
  - Layer 7 (Deduplication Store): Indexes triples with `negated INTEGER`, separating positive from negative assertions.
  - Layer 7 (Semantic Validation): Flags incorrect negation as a failure across validation dimensions.
  - Layer 8 (Publishing): Serializes `negated` flag to Neo4j CSV/Cypher and MeTTa AtomSpace expressions.

---

## 4. Test Commands and Actual Outcomes (Measured)

### Test Run 1: Pre-Extraction Unit Test Suite
- **Command**: `.venv/bin/python -m pytest tests/test_preextraction.py -v --tb=short`
- **Exit Code**: `2` (Collection Error)
- **Actual Output**:
  ```
  ImportError while importing test module 'tests/test_preextraction.py'.
  tests/test_preextraction.py:9: in <module>
      from src.preextraction.preextractor import Preextractor, _merge_entities
  src/preextraction/preextractor.py:5: in <module>
      import spacy
  ModuleNotFoundError: No module named 'spacy'
  ```
- **Root Cause Analysis**:
  1. Primary failure: `spacy` package is missing in `.venv`.
  2. Latent failure: `tests/test_preextraction.py:9` imports `_merge_entities`, but `src/preextraction/preextractor.py` was refactored in commit `80de85c` to use `_resolve_overlaps`. Once `spacy` is resolved, test collection will immediately fail with `ImportError: cannot import name '_merge_entities' from 'src.preextraction.preextractor'`.

### Test Run 2: Integration & Benchmark Suites
- **Commands**:
  - `.venv/bin/python -m pytest tests/test_integration.py`
  - `.venv/bin/python -m pytest tests/test_integration_pipeline.py`
  - `.venv/bin/python -m pytest tests/test_load.py`
- **Actual Outcome**: All three failed collection with the identical error:
  `ModuleNotFoundError: No module named 'spacy'`.

### Test Run 3: Non-Preextraction Test Suite Execution
- **Command**: `.venv/bin/python -m pytest tests/ -v --tb=short --ignore=tests/test_preextraction.py --ignore=tests/test_integration.py --ignore=tests/test_integration_pipeline.py --ignore=tests/test_load.py`
- **Collected**: 250 test items
- **Passed**: 248 passed
- **Failed**: 2 failed
- **Failures Identified**:
  - `tests/test_taxonomy.py::TestCrossSchemaMap::test_all_relation_types_have_cross_schema_entry`: `AssertionError: Missing cross-schema entry for has_inheritance_mode`.
  - `tests/test_taxonomy.py::TestCrossSchemaMap::test_bionlp_field_present`: `KeyError: <RelationType.HAS_INHERITANCE_MODE: 'has_inheritance_mode'>`.
  - **Verdict**: These 2 failures are isolated to `src/schema/taxonomy.py` schema mappings and are completely independent of Layer 4.

---

## 5. Identified Architectural & Algorithmic Deficiencies

Based on static code analysis of the Layer 4 implementation files:

| ID | Component | Defect / Failure Mode | Code Location | Pipeline Impact |
|:---|:---|:---|:---|:---|
| **BUG-1** | `negation_detector.py` | Overwrites NER entity confidence with NLI contradiction score | `negation_detector.py:110` | High-confidence NER predictions (e.g. 0.98) get overwritten with clause contradiction score (e.g. 0.04), destroying confidence provenance. |
| **BUG-2** | `tests/test_preextraction.py` | Stale function import `_merge_entities` | `test_preextraction.py:9` | Test suite cannot collect even after `spacy` is installed. |
| **BUG-3** | `ner_tagger.py` | Deduplication by surface string drops valid distinct mentions | `ner_tagger.py:28-30` | If an entity appears in sentence 1 and again in sentence 3, mention 2 is discarded, breaking position-dependent assertion analysis. |
| **BUG-4** | `pubtator_client.py` | Greedy offset localization binds only to first occurrence | `preextractor.py:157` | If text mentions a gene twice, PubTator annotations always bind to the first offset. |
| **BUG-5** | `hf_ner_tagger.py` | Context length truncation without sliding window | `hf_ner_tagger.py:128` | Passing `text[:8000]` to BERT exceeds the 512 token limit, causing silent truncation or errors. |
| **BUG-6** | `negation_detector.py` | Fixed generic hypothesis causes false positives on baseline/control biology | `negation_detector.py:42-44` | Negative control statements or non-assertive biological descriptions are misclassified as negated entities. |
| **BUG-7** | `negation_detector.py` | Single-clause sequential inference without tensor batching | `negation_detector.py:73-83` | CPU inference runs in an unbatched loop across clauses, introducing significant latency. |

---

## 6. Analysis of Prior Work on `origin/experiment`

A detailed inspection of the remote tracking branch `origin/experiment` reveals substantial prior research conducted by colleagues:

1. **Prior Benchmarking**:
   - `ner/models_experiment/report.md` documents an extensive empirical evaluation of 13 NER architectures across 69 biomedical sentences.
   - **Key Findings**:
     - `GLiNER-BioMed-large` achieved **0.875 Partial F1** and **93.1% Recall**, significantly outperforming scispaCy (0.688 Partial F1, 57.1% Recall).
     - Specialized PubMedBERT models (`BENT-PubMedBERT-Gene`, `BENT-PubMedBERT-Disease`) exhibited very high precision (>90%) but critically low recall (5%–16%) due to narrow benchmark training distribution.
     - `HunFlair2` was deemed unviable for real-time inference due to severe latency (1,188 ms/sentence vs 110 ms for GLiNER).
     - `GLiNER-BioMed-bi-large` crashed due to a version mismatch bug (`BertModel.forward() got an unexpected keyword argument 'token_lengths'`).
2. **Missing Scope on `origin/experiment`**:
   - **Negation detection was never evaluated or improved** on `origin/experiment`.
   - The branch diverged before the overlap resolution refactor in `main` (commit `80de85c`).
   - The `_merge_entities` vs `_resolve_overlaps` discrepancy originated during this divergence.

---

## 7. Limitations & Reproducibility Constraints

- **Dependency Barrier**: Offline test execution of Layer 4 is currently blocked by missing virtual environment wheels (`spacy`, `scispacy`, `transformers`, `torch`, `gliner`).
- **Offline Model Weights**: Neither the NLI model nor the HF NER model is cached on disk. In offline mode (`TRANSFORMERS_OFFLINE=1`), running the baseline produces immediate empty returns or load failures.
- **Gold-Standard Evaluation Set**: The repository does not currently contain an annotated Layer 4 benchmark dataset with ground-truth span labels and negation polarities (only 5 unannotated literature snippet examples in `data/raw_input/sample_literature_snippets.json`).

---

## 8. Proposed Minimal Safe Path for Baseline & Experimentation

To obtain reproducible, measured results without wasting compute or violating environment constraints:

1. **Mock-Verified Unit Testing**: Patch/mock external dependencies in `tests/test_preextraction.py` (matching the technique used in `conftest.py`) so tests run deterministically in the local environment without downloading multi-gigabyte models.
2. **Gold Evaluation Corpus**: Establish a small, validated, 25-sentence biomedical evaluation dataset covering multi-word entities, clinical interventions, contrastive clauses, and explicit negations.
3. **Targeted Bug Fixes**:
   - Fix `_merge_entities` reference in tests to test `_resolve_overlaps`.
   - Fix confidence clobbering in `negation_detector.py`.
   - Update string-only deduplication in `ner_tagger.py` to preserve distinct mention offsets.
4. **Negation Enhancement**: Evaluate rule-enhanced / NegEx contextual filtering to complement or replace the un-cached NLI model.
