# Bio-Semantic Parser

**Bio-Semantic Parser** is a biomedical knowledge graph construction pipeline built at Rejuve.Bio / SingularityNET. It reads scientific papers — PDFs, PubMed, PMC — and extracts structured biological relations (genes, proteins, diseases, drugs and the relations between them) into a Neo4j graph database and a MeTTa AtomSpace.

---

## What it does

Takes scientific papers (PDFs, PubMed, PMC, GEO) as input and produces:
- **Neo4j knowledge graph** — nodes and edges as CSV + Cypher files
- **MeTTa AtomSpace** — structured biological triples
- **Verification reports** — Precision, Recall, F1 per paper

---

## Pipeline — 8 Layers

| Layer | Name | What it does |
|---|---|---|
| 1 | **Registry** | Manages paper sources — `sources.yaml`, APIs, local files |
| 2 | **Scheduler** | Polls sources, deduplicates via SHA256, queues new papers |
| 3 | **Fetcher** | Fetches full text — PMC XML, PubMed, PDFs — splits into 512-token chunks |
| 4 | **Pre-Extraction** | NER tagging, DOI extraction, PubTator3 annotation, negation detection |
| 5 | **Schema / Taxonomy** | 87 relation types + 39 entity types from Biolink, Hetionet, GO, SO |
| 6 | **LLM Extraction** | Gemma 4 extracts triples per chunk, Pydantic validation, 3× self-correction |
| 7 | **Post-Extraction** | Entity normalization → deduplication → cross-chunk linking → two-pass resolution → semantic validation → contradiction detection |
| 8 | **Publish** | Validation gate → Neo4j CSV + MeTTa output → human review queue |

---

## Entity Normalization — 8 Standard Databases

Entity types resolve to canonical IDs using this priority chain:

| Priority | Database | Entity types | ID format |
|---|---|---|---|
| 1 | PubTator3 | All (pre-annotated) | Various |
| 2 | Ensembl REST API | Gene, Transcript | `ENSEMBL:ENSG...` |
| 3 | OLS4 / EBI | All ontology types | `MONDO:`, `GO:`, `CHEBI:` |
| 4 | UniProt | Protein | `P12345` |
| 5 | RxNorm | Small molecule (drugs) | `RXCUI:...` |
| 6 | HMDB | Small molecule (metabolites) | `HMDB:...` |
| 7 | NCBI eSearch | Gene fallback, Variant, Organism | `NCBI_GENE:`, `rs...` |
| 8 | PubChem | Small molecule fallback | `PUBCHEM:...` |
| 9 | Wikidata | Broad fallback | `WD:Q...` |

Gene IDs use **Ensembl by default** to align with the BioCypher approach
(team decision 2026-06-23, replacing NCBI Gene IDs).

---

## Taxonomy — 87 Relation Types

Sourced from four official ontologies:

| Source | Types | Examples |
|---|---|---|
| Biolink Model | 56 | `causes`, `regulates`, `associates_with` |
| Hetionet | 5 | `binds_to`, `expressed_in` |
| Gene Ontology | 19 | PTM relation types |
| Sequence Ontology | 3 | Sequence feature relations |
| Longevity extensions | 7 | Custom Rejuve.Bio types |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add a paper source
vi config/sources.yaml

# Run the pipeline
python tests/demo_showcase.py

# Evaluate accuracy (Precision / Recall / F1)
python tools/evaluate_kg.py --recall
```

---

## Output Structure

```
data/
  kg_output/
    2026-06-23_10-50-04_PDF_6079125b777b/   ← per-paper run (timestamped)
      neo4j/                                ← CSV + Cypher files
      metta/                                ← MeTTa AtomSpace files
      graph.html                            ← interactive graph
      verification_report.html              ← triple verification
      compare_neo4j.html                    ← this paper vs unified KG
    unified_neo4j/                          ← all committed papers combined
    unified_metta/
  triple_store_neo4j.db                     ← unified Neo4j SQLite
  triple_store_metta.db                     ← unified MeTTa SQLite
  checkpoints/                              ← per-layer resumable checkpoints
```

---

## Key Design Decisions

- **Closed taxonomy** — LLM can only use the 87 defined relation types, preventing hallucinated relation labels
- **Ensembl IDs for genes** — aligns with BioCypher approach (team decision 2026-06-23)
- **Human review gate** — triples that fail semantic validation go to `human_review.jsonl` for manual approval before entering the KG
- **Timestamped runs** — each paper processing creates a new folder, previous runs are never overwritten

---

## Documentation

| Doc | Description |
|---|---|
| [docs/todo/future_improvements.md](docs/todo/future_improvements.md) | Planned improvements per layer |
| [docs/architecture/pipeline_overview.svg](docs/architecture/pipeline_overview.svg) | Full pipeline diagram |
| [src/schema/data/sources.md](src/schema/data/sources.md) | Taxonomy source files and download commands |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM extraction | Gemma 4 via vLLM / Ollama (OpenAI-compatible API) |
| Schema validation | Pydantic |
| NER | spaCy + scispaCy |
| Entity annotation | PubTator3 |
| Graph output | Neo4j CSV + MeTTa |
| Frontend | FastAPI + Rich CLI |

---

## NER Model Evaluation Results (scispaCy vs. HunFlair)

To evaluate the impact of different Named Entity Recognition (NER) models on the knowledge graph extraction pipeline, we performed side-by-side benchmarks on **6 PMC articles** using the production fetcher, comparing the existing **scispaCy 4-model ensemble** against **HunFlair2**.

Detailed comparison logs and metrics are stored in [ner_evaluation/results/ner_comparison](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/).

### Overall Evaluation Metrics (Consensus Loop)
Across all 6 benchmarked PMC papers, a total of **2,568 unique entities** were identified:

| Metric / Dimension | scispaCy Ensemble | HunFlair2 |
|---|---|---|
| **Total Entities Found** | 2,405 | 485 |
| **Exclusive Entities** (only found by this model) | 2,083 | 163 |
| **Avg. Overlap Ratio (Agreement)** | 22.2% | 64.9% |
| **Average Span Length (Complexity)** | 12.5 - 15.1 chars | 9.0 - 15.5 chars |

### Per-Paper Detailed Breakdown

| Document ID | Unique (Union) | Consensus | scispaCy (Gain) | HunFlair (Gain) |
|---|---|---|---|---|
| [PMC8160999](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC8160999_20260726_203105_metrics.json) | 149 | 51 | 111 (+60) | 89 (+38) |
| [PMC9500000](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC9500000_20260726_203401_metrics.json) | 286 | 32 | 255 (+223) | 63 (+31) |
| [PMC11097689](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC11097689_20260726_203904_metrics.json) | 210 | 48 | 187 (+139) | 71 (+23) |
| [PMC11975295](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC11975295_20260726_204055_metrics.json) | 199 | 47 | 173 (+126) | 73 (+26) |
| [PMC12659517](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC12659517_20260726_204348_metrics.json) | 1,112 | 35 | 1,098 (+1,063) | 49 (+14) |
| [PMC13083324](file:///home/eyorica/Downloads/Bio-Semantic/bio-semantic-parser/ner_evaluation/results/ner_comparison/PMC13083324_20260726_204626_metrics.json) | 612 | 109 | 581 (+472) | 140 (+31) |

### Key Findings
1. **Precision vs. Recall**: 
   * **scispaCy** acts as a high-recall, high-noise model. It generates a high volume of candidate entities, including a significant amount of noise (such as numbers like `3-ii`, citation references like `[112–114]`, and non-biomedical words).
   * **HunFlair2** is highly conservative and precise. 64.9% of its predictions are confirmed by other models, and its unique predictions (+163 exclusive entities) consist of high-value, precise biomedical entities (e.g. `bcl11b`, `alexa 555`, `ctip2`).
2. **Span Boundary Quality**:
   * HunFlair2 captures tight, normalized spans (e.g. `bcl11b` vs scispaCy's `bcl11b (fig 3`). This directly prevents noisy tokens from entering the Pydantic schema validation step in Layer 6.
3. **Category Strengths**:
   * HunFlair2 shows exceptional performance on `DISEASE` (average F1 of `0.758`) and `ORGANISM` (average F1 of `0.752`).
   * scispaCy is relatively stronger on general coverage and catches various specific `GENE_OR_GENE_PRODUCT` boundary variations.


---

