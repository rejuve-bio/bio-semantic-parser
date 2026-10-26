"""
Layer 4 — Negation Detector

Hybrid contextual negation detector for biomedical literature.
Supports two complementary inference engines:
1. Cross-Encoder NLI: Sequence-pair classifier that scores clauses against an assertive hypothesis.
2. Rule-Based Scope Engine: Deterministic, NegEx/ConText-inspired pattern matcher that tracks
   pre-/post-negation cues, pseudo-negations, scope termination boundaries, and uncertainty.

Engine mode is configurable via NEGATION_ENGINE environment variable:
- "auto" (default): Uses NLI cross-encoder when model weights are available; cleanly falls back
  to rule-based scope detection when running offline without cached weights.
- "nli": Enforces strict NLI inference (raises error if model missing).
- "rules": Uses fast, deterministic rule-based scope engine directly.
"""
from __future__ import annotations
import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# Contrastive conjunctions that split complex sentences into distinct clauses
_CONTRASTIVE = re.compile(
    r'\b(but not|however|although|whereas|while yet|despite|'
    r'in contrast|on the other hand|except|rather than|'
    r'but(?!\s+not)|yet(?!\s+not))\b',
    re.IGNORECASE,
)

# ── NegEx / ConText Biomedical Patterns ──────────────────────────────────────

# Pseudo-negations: phrases containing negation words that do NOT negate findings
_PSEUDO_NEGATION_PATTERNS = [
    re.compile(r'\bno\s+doubt\b', re.I),
    re.compile(r'\bno\s+wonder\b', re.I),
    re.compile(r'\bnot\s+only\b', re.I),
    re.compile(r'\bnot\s+just\b', re.I),
    re.compile(r'\bwithout\s+delay\b', re.I),
    re.compile(r'\bnot\s+necessarily\b', re.I),
    re.compile(r'\bnot\s+certain\b', re.I),
    re.compile(r'\bcannot\s+be\s+ruled\s+out\b', re.I),
    re.compile(r'\bcould\s+not\s+be\s+ruled\s+out\b', re.I),
    re.compile(r'\bnot\s+excluded\b', re.I),
]

# Pre-negation cues: trigger negation forward onto downstream entities within scope
_PRE_NEGATION_PATTERNS = [
    re.compile(r'\bno\s+evidence\s+of\b', re.I),
    re.compile(r'\bno\s+sign\s+of\b', re.I),
    re.compile(r'\bno\s+significant\b', re.I),
    re.compile(r'\bno\s+effect\s+on\b', re.I),
    re.compile(r'\bno\s+improvement\b', re.I),
    re.compile(r'\babsence\s+of\b', re.I),
    re.compile(r'\black\s+of\b', re.I),
    re.compile(r'\bfailure\s+to\b', re.I),
    re.compile(r'\bfailed\s+to\b', re.I),
    re.compile(r'\bfails\s+to\b', re.I),
    re.compile(r'\bunable\s+to\b', re.I),
    re.compile(r'\brules\s+out\b', re.I),
    re.compile(r'\bruled\s+out\b', re.I),
    re.compile(r'\bnegative\s+for\b', re.I),
    re.compile(r'\bdid\s+not\s+show\b', re.I),
    re.compile(r'\bdoes\s+not\s+show\b', re.I),
    re.compile(r'\bdoes\s+not\s+demonstrate\b', re.I),
    re.compile(r'\bdid\s+not\s+demonstrate\b', re.I),
    re.compile(r'\bnot\s+detected\b', re.I),
    re.compile(r'\bnot\s+observed\b', re.I),
    re.compile(r'\bwas\s+not\s+observed\b', re.I),
    re.compile(r'\bwere\s+not\s+observed\b', re.I),
    re.compile(r'\bwas\s+not\s+associated\b', re.I),
    re.compile(r'\bis\s+not\s+associated\b', re.I),
    re.compile(r'\brather\s+than\b', re.I),
    re.compile(r'\bnot\s+recommended\b', re.I),
    re.compile(r'\bdenies\b', re.I),
    re.compile(r'\bdenied\b', re.I),
    re.compile(r'\bno\b', re.I),
    re.compile(r'\bnot\b', re.I),
    re.compile(r'\bneither\b', re.I),
    re.compile(r'\bnor\b', re.I),
    re.compile(r'\bnever\b', re.I),
    re.compile(r'\bwithout\b', re.I),
]

# Post-negation cues: trigger negation backward onto preceding entities within scope
_POST_NEGATION_PATTERNS = [
    re.compile(r'\bwas\s+not\s+seen\b', re.I),
    re.compile(r'\bwere\s+not\s+seen\b', re.I),
    re.compile(r'\bwas\s+absent\b', re.I),
    re.compile(r'\bwere\s+absent\b', re.I),
    re.compile(r'\bwas\s+negative\b', re.I),
    re.compile(r'\bwere\s+negative\b', re.I),
    re.compile(r'\bnot\s+present\b', re.I),
    re.compile(r'\bwas\s+ruled\s+out\b', re.I),
    re.compile(r'\bwere\s+ruled\s+out\b', re.I),
    re.compile(r'\bwas\s+not\s+detected\b', re.I),
    re.compile(r'\bwere\s+not\s+detected\b', re.I),
    re.compile(r'\bwas\s+not\s+expressed\b', re.I),
    re.compile(r'\bwere\s+not\s+expressed\b', re.I),
    re.compile(r'\bnot\s+identified\b', re.I),
    re.compile(r'\bfailed\s+to\s+show\b', re.I),
    re.compile(r'\bfailed\s+to\b', re.I),
    re.compile(r'\b(is|are|was|were)\s+not\s+recommended\b', re.I),
]

# Scope termination markers: boundaries that halt cue propagation
_SCOPE_TERMINATORS = re.compile(
    r'(;|\.|\bbut\b|\bhowever\b|\balthough\b|\bwhereas\b|\bwhile\b|\byet\b|\bexcept\b|'
    r'\bdespite\b|\bin\s+contrast\b|\bon\s+the\s+other\s+hand\b|\brather\s+than\b|'
    r'\binstead\s+of\b|\bwhich\b|\bdue\s+to\b|\bbecause\s+of\b|\bfollowing\b|'
    r'\bfor\s+(chronic|stage|the|patients|treatment)\b|\bin\s+(primary|mice|patients|the|a|human|cells)\b)',
    re.IGNORECASE,
)

# Uncertainty / speculative cues
_UNCERTAINTY_PATTERNS = [
    re.compile(r'\bpossible\b', re.I),
    re.compile(r'\bpossibly\b', re.I),
    re.compile(r'\bpotential\b', re.I),
    re.compile(r'\bpotentially\b', re.I),
    re.compile(r'\bprobable\b', re.I),
    re.compile(r'\bprobably\b', re.I),
    re.compile(r'\bsuspected\b', re.I),
    re.compile(r'\bmay\s+indicate\b', re.I),
    re.compile(r'\bmight\s+suggest\b', re.I),
    re.compile(r'\bunclear\b', re.I),
]

# Historical context cues
_HISTORICAL_PATTERNS = [
    re.compile(r'\bhistory\s+of\b', re.I),
    re.compile(r'\bprior\s+history\b', re.I),
    re.compile(r'\bprevious\s+diagnosis\b', re.I),
]


def _extract_entity_clause(sentence: str, entity_text: str) -> str:
    """Return the sub-clause containing entity_text, or the full sentence if no split point."""
    parts = _CONTRASTIVE.split(sentence)
    if len(parts) <= 1:
        return sentence

    for part in parts:
        if len(part.split()) <= 2:
            continue
        if entity_text.lower() in part.lower():
            stripped = part.strip().strip(",;.")
            if len(stripped) > 10:
                return stripped

    return sentence


def _mask_pseudo_negations(text: str) -> str:
    """Mask pseudo-negation phrases with neutral tokens to avoid false cue triggering."""
    masked = text
    for pat in _PSEUDO_NEGATION_PATTERNS:
        masked = pat.sub(lambda m: "_" * len(m.group(0)), masked)
    return masked


def evaluate_rule_scope(sentence: str, entity_text: str, start_char: int = -1, end_char: int = -1) -> Tuple[bool, float, str]:
    """
    Evaluate negation and assertion for an entity in sentence using NegEx/ConText scoping.
    Returns (is_negated: bool, confidence: float, assertion: str).
    """
    if not sentence or not entity_text:
        return False, 0.05, "PRESENT"

    # Localize entity inside the sentence if offsets not given
    ent_start = start_char
    ent_end = end_char
    if ent_start < 0 or ent_end <= ent_start:
        ent_start = sentence.lower().find(entity_text.lower())
        if ent_start >= 0:
            ent_end = ent_start + len(entity_text)
        else:
            return False, 0.05, "PRESENT"

    prefix = sentence[:ent_start]
    suffix = sentence[ent_end:]
    masked_prefix = _mask_pseudo_negations(prefix)
    masked_suffix = _mask_pseudo_negations(suffix)

    # 1. Check Pre-Negation Scope (Cues preceding entity)
    for pat in _PRE_NEGATION_PATTERNS:
        for match in pat.finditer(masked_prefix):
            cue_end = match.end()
            intervening = prefix[cue_end:]
            # If no termination marker between cue and entity, and within max scope window
            if not _SCOPE_TERMINATORS.search(intervening):
                token_dist = len(intervening.split())
                if token_dist <= 10:  # standard ConText scope window
                    return True, 0.95, "ABSENT"

    # 2. Check Post-Negation Scope (Cues succeeding entity)
    for pat in _POST_NEGATION_PATTERNS:
        for match in pat.finditer(masked_suffix):
            cue_start = match.start()
            intervening = suffix[:cue_start]
            if not _SCOPE_TERMINATORS.search(intervening):
                token_dist = len(intervening.split())
                if token_dist <= 6:
                    return True, 0.95, "ABSENT"

    # 3. Check Uncertainty Scope
    clause = _extract_entity_clause(sentence, entity_text)
    for pat in _UNCERTAINTY_PATTERNS:
        if pat.search(clause):
            return False, 0.15, "POSSIBLE"

    # 4. Check Historical Context
    for pat in _HISTORICAL_PATTERNS:
        if pat.search(clause):
            return False, 0.10, "HISTORICAL"

    return False, 0.05, "PRESENT"


_DEFAULT_HYPOTHESIS = (
    "This study demonstrates a confirmed biological effect, finding, or association."
)


class NegationDetector:
    _MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
    try:
        _THRESHOLD = float(os.getenv("NEGATION_THRESHOLD", "0.45"))
    except ValueError:
        _THRESHOLD = 0.45

    def __init__(self, hypothesis: str = _DEFAULT_HYPOTHESIS):
        self._hypothesis = hypothesis
        self._engine_mode = os.getenv("NEGATION_ENGINE", "auto").lower()
        self._model = None
        self._tokenizer = None
        self._model_loaded = False

        if self._engine_mode in ("auto", "nli"):
            try:
                import os as _os
                _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                if AutoTokenizer is None or AutoModelForSequenceClassification is None:
                    raise ImportError("transformers not installed")

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._MODEL, local_files_only=True
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._MODEL,
                    local_files_only=True,
                    device_map=None,          # prevent accelerate meta tensor path
                    low_cpu_mem_usage=False,  # force eager allocation
                )
                self._model.eval()
                self._model_loaded = True
            except Exception as e:
                if self._engine_mode == "nli":
                    raise
                # Fallback cleanly in 'auto' mode
                logger.info(
                    "Negation NLI model not available locally (%s); using rule-based scope engine.", e
                )
                self._model_loaded = False

    def _is_negated(self, clause: str) -> tuple:
        """Score clause contradiction against hypothesis using NLI or rule-based fallback."""
        if self._model_loaded and self._tokenizer is not None and self._model is not None:
            try:
                import torch
                inputs = self._tokenizer(
                    clause, self._hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    probs = torch.softmax(self._model(**inputs).logits, dim=-1)[0]
                contradiction_score = float(probs[0])
                return contradiction_score > self._THRESHOLD, round(contradiction_score, 3)
            except Exception:
                pass  # Fall through to rule-based fallback on tensor error

        # Rule-based fallback
        is_neg, conf, _ = evaluate_rule_scope(clause, clause)
        return is_neg, conf

    def _find_sentence(self, doc, entity_text: str, start_char: int) -> tuple[str, int, int]:
        """Find the containing sentence and the entity's relative offsets within that sentence."""
        if hasattr(doc, "sents"):
            if start_char >= 0:
                for sent in doc.sents:
                    if sent.start_char <= start_char < sent.end_char:
                        rel_start = start_char - sent.start_char
                        rel_end = rel_start + len(entity_text)
                        return sent.text, rel_start, rel_end
            # PubTator3 entities or missing char offsets — fallback to text search across sents
            for sent in doc.sents:
                idx = sent.text.lower().find(entity_text.lower())
                if idx >= 0:
                    return sent.text, idx, idx + len(entity_text)

        doc_text = getattr(doc, "text", str(doc))
        idx = doc_text.lower().find(entity_text.lower())
        rel_start = idx if idx >= 0 else -1
        rel_end = (idx + len(entity_text)) if idx >= 0 else -1
        return doc_text, rel_start, rel_end

    def process(self, entities: list, doc) -> dict:
        clause_cache: dict = {}
        updated = []
        negated = []

        for ent in entities:
            sentence, rel_start, rel_end = self._find_sentence(doc, ent["text"], ent.get("start", -1))
            clause = _extract_entity_clause(sentence, ent["text"])

            # 1. Rule-based scope evaluation for the entity in its sentence
            rule_neg, rule_conf, rule_assertion = evaluate_rule_scope(
                sentence, ent["text"], rel_start, rel_end
            )

            # 2. Model evaluation when NLI is available
            if self._model_loaded:
                if clause not in clause_cache:
                    clause_cache[clause] = self._is_negated(clause)
                nli_neg, nli_conf = clause_cache[clause]

                # Combine: NLI contradiction confirmed by or refined by rule scope
                # If rule explicitly identified a clear pre/post negation cue within scope, trust it
                if rule_neg:
                    is_neg = True
                    neg_conf = max(nli_conf, rule_conf)
                    assertion = "ABSENT"
                elif nli_neg and not rule_neg:
                    # Check if clause has contrastive structure that may mislead NLI
                    if any(t in sentence.lower() for t in ["but not", "however", "whereas"]):
                        # In contrastive sentences, entity outside negation scope should NOT be negated
                        is_neg = rule_neg
                        neg_conf = rule_conf
                        assertion = rule_assertion
                    else:
                        is_neg = True
                        neg_conf = nli_conf
                        assertion = "ABSENT"
                else:
                    is_neg = False
                    neg_conf = nli_conf
                    assertion = rule_assertion
            else:
                is_neg = rule_neg
                neg_conf = rule_conf
                assertion = rule_assertion

            # CRITICAL (BUG-1 Fix): Preserve original NER entity detection confidence!
            # Do NOT clobber ent["confidence"] with the negation score.
            orig_ner_conf = ent.get("ner_confidence", ent.get("confidence", 1.0))

            entry = {
                **ent,
                "negated":             is_neg,
                "assertion":           assertion,
                "confidence":          orig_ner_conf,
                "ner_confidence":      orig_ner_conf,
                "negation_confidence": round(neg_conf, 3),
            }
            updated.append(entry)
            if is_neg:
                negated.append(entry)

        return {
            "entities":         updated,
            "has_negation":     bool(negated),
            "negated_entities": negated,
        }
