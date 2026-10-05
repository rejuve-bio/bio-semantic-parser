"""
Layer 4 — Scope-Based Negation / Speculation Detector

Uses a fine-tuned PubMedBERT token classifier to predict negation and
speculation scopes over full sentences, then maps those token-level scopes
back onto existing NER entity spans.

This detector is intentionally sentence-based: the fine-tuned model was
trained on full-sentence scope annotations, so it does not require the
contrastive clause splitting used by the NLI detector.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


_BACKBONE = os.getenv(
    "SCOPE_NEGATION_BACKBONE",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
)
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "best_model.pt"
)
# PubMedBERT supports up to 512 tokens. 128 silently truncates long biomedical
# sentences (often 60–100+ words), losing scope labels for words beyond position
# ~100.  We default to 512 and let users lower it for memory-constrained envs.
_MAX_LEN = int(os.getenv("SCOPE_NEGATION_MAX_LEN", "512"))
_NEG_THRESHOLD = float(os.getenv("SCOPE_NEGATION_OVERLAP_THRESHOLD", "0.50"))
_SPEC_THRESHOLD = float(os.getenv("SCOPE_SPECULATION_OVERLAP_THRESHOLD", "0.50"))

_tokenizer = None
_model = None
_model_path = ""


import threading

_lock = threading.Lock()

class _MultiHeadTokenClassifier(nn.Module):
    def __init__(self, backbone_name: str, dropout: float = 0.15, num_labels: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            local_files_only=True,
            device_map=None,          
            low_cpu_mem_usage=False, 
        )
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_neg = nn.Linear(hidden, num_labels)
        self.head_spec = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.dropout(x)
        return self.head_neg(x), self.head_spec(x)


def should_run() -> bool:
    return os.getenv("SCOPE_NEGATION_ENABLED", "false").lower() == "true"


def _resolve_model_path() -> Path:
    custom = os.getenv("SCOPE_NEGATION_MODEL_PATH", "").strip()
    return Path(custom) if custom else _DEFAULT_MODEL_PATH


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_components():
    global _tokenizer, _model, _model_path

    model_path = str(_resolve_model_path())
    if _tokenizer is not None and _model is not None and _model_path == model_path:
        return _tokenizer, _model

    with _lock:
        if _tokenizer is not None and _model is not None and _model_path == model_path:
            return _tokenizer, _model

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ScopeNegation] Model weights file not found at: {model_path}. "
                f"Ensure best_model.pt is placed in the project root or set SCOPE_NEGATION_MODEL_PATH."
            )

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        tokenizer = AutoTokenizer.from_pretrained(_BACKBONE, local_files_only=True)
        model = _MultiHeadTokenClassifier(_BACKBONE)
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.to(_get_device())
        model.eval()

        _tokenizer = tokenizer
        _model = model
        _model_path = model_path
        return _tokenizer, _model


# ---------------------------------------------------------------------------
# Span utilities
# ---------------------------------------------------------------------------

def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping / touching char spans into a sorted, disjoint list.

    This is needed before computing _overlap_ratio so that overlapping scope
    spans (which can occur when adjacent subword tokens both get label=1 and
    their character windows touch) are not double-counted.
    """
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged: list[tuple[int, int]] = [sorted_spans[0]]
    for s0, s1 in sorted_spans[1:]:
        prev_s0, prev_s1 = merged[-1]
        if s0 <= prev_s1:          # overlapping or touching
            merged[-1] = (prev_s0, max(prev_s1, s1))
        else:
            merged.append((s0, s1))
    return merged


def _overlap_ratio(start: int, end: int, spans: list[tuple[int, int]]) -> float:
    """Fraction of the [start, end) entity window covered by the union of spans.

    Spans are merged first so overlapping scope tokens are not double-counted.
    """
    if start < 0 or end <= start or not spans:
        return 0.0
    total = end - start
    merged = _merge_spans(spans)
    covered = sum(
        max(0, min(end, s1) - max(start, s0))
        for s0, s1 in merged
    )
    return covered / total


# ---------------------------------------------------------------------------
# Sentence-level scope prediction
# ---------------------------------------------------------------------------

def _build_word_char_offsets(sentence: str) -> list[tuple[int, int]]:
    """Return character (start, end) for each whitespace-split word.

    Unlike ``sentence.find(word, pos)``, this method tracks the exact byte
    position as we consume the string left-to-right, so it is correct even
    when the same token substring appears in multiple words (e.g. 'cell'
    inside 'T-cell' and later as standalone 'cell').
    """
    offsets: list[tuple[int, int]] = []
    pos = 0
    for word in sentence.split():
        # Skip any whitespace before this word
        while pos < len(sentence) and sentence[pos].isspace():
            pos += 1
        # The word must start at pos (split() guarantees this)
        start = pos
        end = start + len(word)
        offsets.append((start, end))
        pos = end
    return offsets


def _predict_sentence_scopes(sentence: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Run the scope model on one sentence and return (neg_spans, spec_spans).

    Each span is a (char_start, char_end) tuple in sentence-local coordinates.
    Returned span lists are sorted and contain disjoint intervals.
    """
    tokenizer, model = _load_components()
    device = next(model.parameters()).device

    words = sentence.split()
    if not words:
        return [], []

    # Build accurate character offsets before tokenising so we never rely on
    # find() to reconstruct positions after the fact.
    word_char_offsets = _build_word_char_offsets(sentence)

    # Subword tokenisation — track which word each subword belongs to.
    input_ids: list[int] = [tokenizer.cls_token_id]
    word_of_subword: list[Optional[int]] = [None]
    for w_idx, word in enumerate(words):
        subwords = tokenizer.tokenize(word)
        if not subwords:
            continue
        input_ids.extend(tokenizer.convert_tokens_to_ids(subwords))
        word_of_subword.extend([w_idx] * len(subwords))

    input_ids      = input_ids[:_MAX_LEN - 1] + [tokenizer.sep_token_id]
    word_of_subword = word_of_subword[:_MAX_LEN - 1] + [None]
    attention_mask  = [1] * len(input_ids)

    pad_len = _MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids       += [tokenizer.pad_token_id] * pad_len
        attention_mask  += [0] * pad_len
        word_of_subword += [None] * pad_len

    ids_t  = torch.tensor([input_ids],      dtype=torch.long, device=device)
    mask_t = torch.tensor([attention_mask],  dtype=torch.long, device=device)

    with torch.no_grad():
        logits_neg, logits_spec = model(ids_t, mask_t)
        pred_neg  = logits_neg.argmax(dim=-1).squeeze(0).cpu().tolist()
        pred_spec = logits_spec.argmax(dim=-1).squeeze(0).cpu().tolist()

    # Aggregate subword labels to word labels (any-subword-positive → word positive)
    word_neg  = [0] * len(words)
    word_spec = [0] * len(words)
    for sub_idx, w_idx in enumerate(word_of_subword):
        if w_idx is None:
            continue
        if pred_neg[sub_idx] == 1:
            word_neg[w_idx] = 1
        if pred_spec[sub_idx] == 1:
            word_spec[w_idx] = 1

    # Build span lists using pre-computed exact character offsets.
    neg_spans:  list[tuple[int, int]] = []
    spec_spans: list[tuple[int, int]] = []
    for i, (char_start, char_end) in enumerate(word_char_offsets):
        if word_neg[i]:
            neg_spans.append((char_start, char_end))
        if word_spec[i]:
            spec_spans.append((char_start, char_end))

    # Merge touching / overlapping adjacent scope spans into contiguous blocks.
    # This makes overlap_ratio behave correctly for entity spans that straddle
    # multiple consecutive scope-labelled words.
    return _merge_spans(neg_spans), _merge_spans(spec_spans)


def _find_sentence(doc, entity_text: str, start_char: int):
    if start_char >= 0:
        for sent in doc.sents:
            if sent.start_char <= start_char < sent.end_char:
                return sent
    for sent in doc.sents:
        if entity_text.lower() in sent.text.lower():
            return sent
    return doc[:]


class ScopeNegationDetector:
    """Map token-level scope predictions back onto entity spans."""

    def process(self, entities: list, doc) -> dict:
        sentence_cache: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
        updated = []
        negated = []
        speculative = []

        for ent in entities:
            sent = _find_sentence(doc, ent["text"], ent.get("start", -1))
            sentence = sent.text
            if sentence not in sentence_cache:
                sentence_cache[sentence] = _predict_sentence_scopes(sentence)
            neg_spans, spec_spans = sentence_cache[sentence]

            if ent.get("start", -1) >= 0 and ent.get("end", -1) > ent.get("start", -1):
                local_start = ent["start"] - sent.start_char
                local_end   = ent["end"]   - sent.start_char
            else:
                local_start = sentence.lower().find(ent["text"].lower())
                local_end   = local_start + len(ent["text"]) if local_start >= 0 else -1

            neg_overlap  = _overlap_ratio(local_start, local_end, neg_spans)
            spec_overlap = _overlap_ratio(local_start, local_end, spec_spans)
            is_neg  = neg_overlap  >= _NEG_THRESHOLD
            is_spec = spec_overlap >= _SPEC_THRESHOLD

            assertion = "ABSENT" if is_neg else "POSSIBLE" if is_spec else "PRESENT"
            entry = {
                **ent,
                "negated":                   is_neg,
                "speculative":               is_spec,
                "assertion":                 assertion,
                "confidence":                round(max(neg_overlap, spec_overlap, ent.get("confidence", 0.0)), 3),
                "negation_source":           "scope_model",
                "negation_scope_overlap":    round(neg_overlap,  3),
                "speculation_scope_overlap": round(spec_overlap, 3),
            }
            updated.append(entry)
            if is_neg:
                negated.append(entry)
            if is_spec:
                speculative.append(entry)

        return {
            "entities":             updated,
            "has_negation":         bool(negated),
            "negated_entities":     negated,
            "speculative_entities": speculative,
        }
