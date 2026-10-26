"""
src/preextraction/hf_ner_tagger.py
────────────────────────────────────
HuggingFace transformer-based NER tagger — supplements the scispaCy ensemble
with clinical entity types that no scispaCy model covers:

  Diagnostic_procedure  → PROCEDURE / CLINICAL_INTERVENTION
  Therapeutic_procedure → CLINICAL_INTERVENTION / TREATMENT
  Medication            → DRUG
  Sign_symptom          → SYMPTOM
  Disease_disorder      → DISEASE
  Lab_value             → CLINICAL_MEASUREMENT

Model is configurable via .env:
    HF_NER_MODEL=d4data/biomedical-ner-all   (default)
    HF_NER_ENABLED=true/false                (default: true)
    HF_NER_THRESHOLD=0.80                    (min confidence score)

Label mapping from model output → our EntityType taxonomy is defined in
HF_LABEL_MAP below — extend it if you add a different model.
"""
from __future__ import annotations
import os
from typing import Optional

# Map model-specific labels → our EntityType taxonomy values
# Extend this dict when switching to a different HuggingFace model.
HF_LABEL_MAP: dict[str, str] = {
    # d4data/biomedical-ner-all labels
    "Diagnostic_procedure":  "PROCEDURE",
    "Therapeutic_procedure": "CLINICAL_INTERVENTION",
    "Medication":            "DRUG",
    "Sign_symptom":          "SYMPTOM",
    "Disease_disorder":      "DISEASE",
    "Lab_value":             "CLINICAL_MEASUREMENT",
    "Detailed_description":  "OTHER",
    "Activity":              "ACTIVITY",
    "Biological_structure":  "ANATOMY",
    "Clinical_event":        "EVENT",
    "Coreference":           "OTHER",
    "Date":                  "OTHER",
    "Dosage":                "OTHER",
    "Duration":              "OTHER",
    "Family_history":        "OTHER",
    "Frequency":             "OTHER",
    "History":               "OTHER",
    "Nonbiological_location":"OTHER",
    "Other_entity":          "OTHER",
    "Other_event":           "EVENT",
    "Personal_background":   "OTHER",
    "Qualitative_concept":   "OTHER",
    "Quantitative_concept":  "OTHER",
    "Severity":              "OTHER",
    "Time":                  "OTHER",
    "Volume":                "OTHER",
    "Weight":                "OTHER",
}

_pipeline: Optional[object] = None
_pipeline_model: str = ""
import threading as _threading
_pipeline_lock = _threading.Lock()


def _get_pipeline():
    """Lazy-load the HuggingFace NER pipeline (cached after first call)."""
    global _pipeline, _pipeline_model
    model_name = os.getenv("HF_NER_MODEL", "d4data/biomedical-ner-all")
    if _pipeline is None or _pipeline_model != model_name:
        import torch
        from transformers import pipeline as _hf_pipeline, AutoTokenizer, AutoModelForTokenClassification

        # Offline mode — use cached models, no network checks
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"]  = "1"

        # Load explicitly without device_map and without accelerate meta tensors.
        # device_map=None + low_cpu_mem_usage=False forces eager (non-meta) loading.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            device_map=None,           # never use accelerate's meta tensor path
            low_cpu_mem_usage=False,   # force full eager allocation
            local_files_only=True,
        )

        # Pick best available device
        if torch.cuda.is_available():
            device = 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1   # CPU

        # Ensure tokenizer model_max_length is realistically bounded to model's positional embeddings
        max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 512)
        if not isinstance(max_pos, int) or max_pos <= 0:
            max_pos = 512
        if hasattr(tokenizer, "model_max_length"):
            if not isinstance(tokenizer.model_max_length, int) or tokenizer.model_max_length > max_pos:
                tokenizer.model_max_length = max_pos

        new_pipeline = _hf_pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        with _pipeline_lock:
            _pipeline = new_pipeline
            _pipeline_model = model_name
    return _pipeline


def should_run() -> bool:
    return os.getenv("HF_NER_ENABLED", "true").lower() == "true"


def _split_text_windows(text: str, max_chars: int = 1500) -> list[tuple[int, str]]:
    r"""Split text into sentence-aware windows <= max_chars to stay within model token limits.
    Windows are contiguous exact substring slices of text, preserving pixel-perfect character offsets.
    Priority order for split points:
      1. Sentence boundaries ((?<=[.!?\n])\s+)
      2. Word boundaries (\s+)
      3. Hard character slice at max_chars
    Guarantees:
      - len(w_text) <= max_chars for every returned window
      - text[w_offset : w_offset + len(w_text)] == w_text
      - "".join(w_text for _, w_text in windows) == text
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [(0, text)]

    windows = []
    w_start = 0
    n = len(text)
    import re

    sent_boundary = re.compile(r'(?<=[.!?\n])\s+')
    word_boundary = re.compile(r'\s+')

    while w_start < n:
        remaining = n - w_start
        if remaining <= max_chars:
            windows.append((w_start, text[w_start:]))
            break

        target_end = w_start + max_chars
        candidate = text[w_start:target_end]

        # 1. Prefer sentence boundary
        split_pos = -1
        for m in sent_boundary.finditer(candidate):
            if m.end() > 0:
                split_pos = m.end()

        # 2. Fall back to word boundary within window
        if split_pos <= 0:
            for m in word_boundary.finditer(candidate):
                if m.end() > 0:
                    split_pos = m.end()

        # 3. Fall back to hard slice at max_chars for unbroken text
        if split_pos <= 0:
            split_pos = max_chars

        w_end = w_start + split_pos
        windows.append((w_start, text[w_start:w_end]))
        w_start = w_end

    return windows


def tag_entities(text: str) -> list[dict]:
    """
    Run the HuggingFace NER model on text using sentence-aware windows.
    Returns entity dicts in the same format as NERTagger.from_doc().
    Only entities with mapped labels (not OTHER) and score >= HF_NER_THRESHOLD
    are returned.
    """
    if not should_run() or not text:
        return []

    threshold = float(os.getenv("HF_NER_THRESHOLD", "0.80"))
    try:
        max_chars = int(os.getenv("HF_NER_MAX_CHARS", "1500"))
    except ValueError:
        max_chars = 1500

    try:
        pipe = _get_pipeline()
    except Exception:
        return []

    entities = []
    seen = set()

    for w_offset, w_text in _split_text_windows(text, max_chars=max_chars):
        try:
            results = pipe(w_text)
        except Exception:
            try:
                results = pipe(w_text, truncation=True)
            except Exception:
                continue

        for r in results:
            word  = r.get("word", "").strip()
            label = r.get("entity_group", "")
            score = float(r.get("score", 0.0))

            if not word or len(word) < 2:
                continue
            if score < threshold:
                continue

            entity_type = HF_LABEL_MAP.get(label, "OTHER")
            if entity_type == "OTHER":
                continue

            # Clean BERT subword tokenization artefacts
            word = word.replace(" ##", "").replace("##", "").strip()
            if not word or len(word) <= 2:
                continue
            if word.lower() in {"the", "a", "an", "of", "in", "to", "is", "are", "was",
                                 "low", "high", "new", "old", "non", "pre", "pro", "anti"}:
                continue

            local_start = int(r.get("start", -1))
            local_end   = int(r.get("end", -1))
            if local_start >= 0 and local_end > local_start:
                start = w_offset + local_start
                end   = w_offset + local_end
                if 0 <= start < end <= len(text):
                    word = text[start:end]
                else:
                    start = text.lower().find(word.lower(), w_offset)
                    end   = start + len(word) if start >= 0 else -1
            else:
                word_norm = word.replace(" - ", "-").replace("- ", "-").replace(" -", "-")
                start = text.lower().find(word_norm.lower(), w_offset)
                if start < 0:
                    start = text.lower().find(word.lower(), w_offset)
                end = start + len(word) if start >= 0 else -1

            if start < 0 or word.lower() not in text.lower():
                continue

            key = word.lower()
            # Deduplicate by span to preserve repeated mentions at distinct offsets
            span_key = (key, start, end)
            if span_key in seen:
                continue
            seen.add(span_key)

            entities.append({
                "text":           word,
                "normalized":     key,
                "label":          entity_type,
                "start":          start,
                "end":            end,
                "negated":        False,
                "assertion":      "PRESENT",
                "confidence":     round(score, 3),
                "ner_confidence": round(score, 3),
                "source":         "hf_ner",
            })

    return entities
