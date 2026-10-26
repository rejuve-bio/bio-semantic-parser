"""
Layer 4 — NER Tagger

Converts spaCy Doc entities into the pipeline's standard entity dict format.
Applies a span quality filter to drop statistical notation, bracket fragments,
and numeric-only spans.
"""


class NERTagger:
    @staticmethod
    def _is_valid(text: str) -> bool:
        """At least 30% alphabetic characters; must not start with a bracket."""
        if len(text.strip()) < 2:
            return False
        if text[0] in ('[', '('):
            return False
        return sum(c.isalpha() for c in text) / len(text) >= 0.30

    @staticmethod
    def from_doc(doc) -> list:
        seen     = set()
        entities = []
        for ent in doc.ents:
            if not NERTagger._is_valid(ent.text):
                continue
            normalized = ent.text.lower().strip()
            # Deduplicate only identical spans at the same character offsets
            start_c = getattr(ent, "start_char", -1)
            end_c   = getattr(ent, "end_char", -1)
            dedup_key = (normalized, start_c, end_c) if start_c >= 0 else normalized
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            score = 1.0
            source = "ensemble"
            # Support spaCy span custom extensions if registered and set to numeric score
            if hasattr(ent, "_"):
                val = getattr(ent._, "score", None)
                if isinstance(val, (int, float)):
                    score = float(val)
                src_val = getattr(ent._, "source", None)
                if isinstance(src_val, str) and src_val:
                    source = src_val

            entities.append({
                "text":           ent.text,
                "normalized":     normalized,
                "label":          ent.label_,
                "start":          start_c,
                "end":            end_c,
                "negated":        False,
                "assertion":      "PRESENT",
                "confidence":     score,
                "ner_confidence": score,
                "source":         source,
            })
        return entities
