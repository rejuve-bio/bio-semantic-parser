#!/usr/bin/env python3
"""
Compare the production NLI negation detector against the optional scope model
on a real PMC paper fetched through Layer 3 production logic.

Usage:
    .venv/bin/python negation_experiment/scripts/compare_negation_pmc.py PMC8160999
    .venv/bin/python negation_experiment/scripts/compare_negation_pmc.py PMC11975295 --fetch
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.fetcher.fetcher import Fetcher
from src.preextraction.ner_tagger import NERTagger
from src.preextraction.negation_detector import NegationDetector, _extract_entity_clause
from src.preextraction.preextractor import Preextractor
from src.preextraction.scope_negation_detector import ScopeNegationDetector

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _fetch_europepmc_chunks(pmc_id: str) -> list[dict]:
    """Fallback fetcher when the production Fetcher/coref service is unavailable."""
    resp = urllib.request.urlopen(
        f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id}/fullTextXML",
        timeout=60,
    )
    root = ET.fromstring(resp.read())
    paragraphs = [
        "".join(p.itertext()).strip()
        for p in root.iter("p")
        if len("".join(p.itertext()).strip()) > 30
    ]
    full_text = re.sub(r"\[\d+(?:[,\-–]\s*\d+)*\]", "", " ".join(paragraphs))
    words = full_text.split()
    chunk_size = 350
    chunks: list[dict] = []
    for i in range(0, len(words), chunk_size):
        chunks.append({
            "text": " ".join(words[i:i + chunk_size]),
            "section": "body",
            "chunk_index": len(chunks),
            "document_id": pmc_id,
            "source_name": "pmc",
        })
    return chunks


def _fetch_pmc_chunks(pmc_id: str, force_fetch: bool = False) -> tuple[list[dict], str]:
    proj_root = _PROJ_ROOT
    ckpt_file = proj_root / "data" / "checkpoints" / pmc_id / "layer3_chunks.json"
    if ckpt_file.exists() and not force_fetch:
        chunks = json.loads(ckpt_file.read_text(encoding="utf-8"))
        return chunks, "checkpoint"

    num_id = pmc_id.upper().lstrip("PMC")
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={num_id}&rettype=full&retmode=xml"
    )
    paper_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
    source = {"name": "pmc", "type": "xml", "rate_limit_delay": 0.34}

    coref_url = os.getenv("COREF_SERVICE_URL", "").strip() or "http://dummy:9999"
    fetcher = Fetcher(coref_url=coref_url)
    fetcher.coref_client.health_check = lambda: True
    try:
        chunks = fetcher.fetch(
            url=fetch_url,
            source=source,
            document_id=pmc_id,
            paper_url=paper_url,
            verbose=False,
        )
        if chunks and any(c.get("text", "").strip() for c in chunks):
            return chunks, "fetcher"
    except Exception:
        pass

    chunks = _fetch_europepmc_chunks(pmc_id)
    return chunks, "europepmc"


def _highlight_entity(text: str, entity_text: str, start: int = -1, end: int = -1) -> str:
    if start >= 0 and end > start and text[start:end].lower() == (entity_text or "").lower():
        return f"{text[:start]}[[{text[start:end]}]]{text[end:]}"
    idx = text.lower().find((entity_text or "").lower())
    if idx < 0:
        return text
    return f"{text[:idx]}[[{text[idx:idx + len(entity_text)]}]]{text[idx + len(entity_text):]}"


def _paragraph_excerpt(chunk_text: str, start: int, end: int, radius: int = 220) -> str:
    if start < 0 or end <= start:
        return chunk_text[: min(len(chunk_text), radius * 2)]
    left = max(0, start - radius)
    right = min(len(chunk_text), end + radius)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(chunk_text) else ""
    return prefix + chunk_text[left:right] + suffix


def _entity_context(
    doc,
    nli_detector: NegationDetector,
    ent: dict,
    chunk_text: str,
) -> dict:
    sentence = nli_detector._find_sentence(doc, ent.get("text", ""), ent.get("start", -1))
    clause = _extract_entity_clause(sentence, ent.get("text", ""))
    start = ent.get("start", -1)
    end = ent.get("end", -1)
    return {
        "sentence": sentence,
        "nli_clause": clause,
        "paragraph_excerpt": _paragraph_excerpt(chunk_text, start, end),
        "highlighted_sentence": _highlight_entity(sentence, ent.get("text", ""), start=-1, end=-1),
        "highlighted_clause": _highlight_entity(clause, ent.get("text", ""), start=-1, end=-1),
        "highlighted_paragraph": _highlight_entity(
            _paragraph_excerpt(chunk_text, start, end),
            ent.get("text", ""),
            start=-1,
            end=-1,
        ),
    }


def _compare_entities(
    nli_entities: list[dict],
    scope_entities: list[dict],
    doc,
    nli_detector: NegationDetector,
    chunk_text: str,
) -> dict:
    scope_by_key = {
        ((e.get("text") or "").lower(), e.get("start", -1), e.get("end", -1)): e
        for e in scope_entities
    }
    rows = []
    for ent in nli_entities:
        key = ((ent.get("text") or "").lower(), ent.get("start", -1), ent.get("end", -1))
        scope_ent = scope_by_key.get(key, {})
        context = _entity_context(doc, nli_detector, ent, chunk_text)
        rows.append({
            "text": ent.get("text"),
            "label": ent.get("label"),
            "start": ent.get("start", -1),
            "end": ent.get("end", -1),
            "nli_negated": bool(ent.get("negated")),
            "nli_confidence": ent.get("confidence", 0.0),
            "scope_negated": bool(scope_ent.get("negated", False)),
            "scope_speculative": bool(scope_ent.get("speculative", False)),
            "scope_excluded": bool(scope_ent.get("scope_excluded", False)),
            "scope_negation_type": scope_ent.get("scope_negation_type", "none"),
            "scope_confidence": scope_ent.get("confidence", 0.0),
            "negation_scope_overlap": scope_ent.get("negation_scope_overlap", 0.0),
            "speculation_scope_overlap": scope_ent.get("speculation_scope_overlap", 0.0),
            **context,
        })

    disagreements = [r for r in rows if r["nli_negated"] != r["scope_negated"]]
    return {
        "rows": rows,
        "agreement_count": len(rows) - len(disagreements),
        "disagreement_count": len(disagreements),
        "nli_negated_count": sum(1 for r in rows if r["nli_negated"]),
        "scope_negated_count": sum(1 for r in rows if r["scope_negated"]),
        "scope_speculative_count": sum(1 for r in rows if r["scope_speculative"]),
        "scope_excluded_count": sum(1 for r in rows if r["scope_excluded"]),
        "disagreements": disagreements,
        "nli_negated_examples": [r for r in rows if r["nli_negated"]],
        "scope_negated_examples": [r for r in rows if r["scope_negated"]],
        "scope_speculative_examples": [r for r in rows if r["scope_speculative"]],
        "scope_excluded_examples": [r for r in rows if r["scope_excluded"]],
    }


def _print_example(title: str, row: dict) -> None:
    print(f"\n  {title}: {row['text']} ({row['label']})")
    print(
        f"    NLI neg={row['nli_negated']} ({row['nli_confidence']}) | "
        f"Scope type={row.get('scope_negation_type', 'none')} "
        f"neg={row['scope_negated']} spec={row['scope_speculative']} "
        f"excluded={row.get('scope_excluded', False)} "
        f"overlap={row['negation_scope_overlap']}/{row['speculation_scope_overlap']}"
    )
    print(f"    Sentence: {row['highlighted_sentence']}")
    if row["nli_clause"] != row["sentence"]:
        print(f"    NLI clause: {row['highlighted_clause']}")
    print(f"    Paragraph: {row['highlighted_paragraph']}")


def _write_markdown_report(path: Path, report: dict) -> None:
  lines = [
      f"# Negation comparison — {report['pmc_id']}",
      "",
      f"- Source: `{report['source']}`",
      f"- Entities compared: **{report['entity_count']}**",
      f"- Agreement: **{report['agreement_count']}/{report['entity_count']}**",
      f"- NLI negated: **{report['nli_negated_count']}**",
      f"- Scope negated: **{report['scope_negated_count']}**",
      f"- Scope speculative: **{report['scope_speculative_count']}**",
      f"- Scope excluded: **{report.get('scope_excluded_count', 0)}**",
      "",
  ]

  def _section(title: str, items: list[dict], limit: int = 20) -> None:
      lines.append(f"## {title}")
      lines.append("")
      if not items:
          lines.append("_None_")
          lines.append("")
          return
      for row in items[:limit]:
          lines.append(f"### {row['text']} ({row['label']})")
          lines.append(
              f"- NLI negated: `{row['nli_negated']}` ({row['nli_confidence']})"
          )
          lines.append(
              f"- Scope negated: `{row['scope_negated']}` | "
              f"Scope speculative: `{row['scope_speculative']}` | "
              f"Scope excluded: `{row.get('scope_excluded', False)}` | "
              f"Type: `{row.get('scope_negation_type', 'none')}`"
          )
          lines.append(f"- Sentence: {row['highlighted_sentence']}")
          if row["nli_clause"] != row["sentence"]:
              lines.append(f"- NLI clause: {row['highlighted_clause']}")
          lines.append(f"- Paragraph: {row['highlighted_paragraph']}")
          lines.append("")

  all_rows = [row for chunk in report["chunks"] for row in chunk.get("rows", [])]
  _section("NLI negated examples", [r for r in all_rows if r["nli_negated"]])
  _section("Scope negated examples", [r for r in all_rows if r["scope_negated"]])
  _section("Scope speculative examples", [r for r in all_rows if r["scope_speculative"]])
  _section("Scope excluded examples", [r for r in all_rows if r.get("scope_excluded")])
  _section("Disagreements", [r for r in all_rows if r["nli_negated"] != r["scope_negated"]])

  path.write_text("\n".join(lines), encoding="utf-8")


def main():
    pmc_id = sys.argv[1] if len(sys.argv) > 1 else "PMC8160999"
    force_fetch = "--fetch" in sys.argv

    preextractor = Preextractor()
    nli_detector = NegationDetector()
    scope_detector = ScopeNegationDetector()

    chunks, source = _fetch_pmc_chunks(pmc_id, force_fetch=force_fetch)
    if not chunks:
        raise SystemExit(f"No chunks available for {pmc_id}")

    report_chunks = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        if not text.strip():
            continue
        doc = preextractor._run_ensemble(text)
        entities = NERTagger.from_doc(doc)
        nli = nli_detector.process(entities, doc)
        scope = scope_detector.process(entities, doc)
        cmp = _compare_entities(
            nli["entities"],
            scope["entities"],
            doc,
            nli_detector,
            text,
        )

        report_chunks.append({
            "chunk_index": idx - 1,
            "section": chunk.get("section", "?"),
            "chars": len(text),
            "paragraph_preview": text[:500] + ("…" if len(text) > 500 else ""),
            "entity_count": len(cmp["rows"]),
            "agreement_count": cmp["agreement_count"],
            "disagreement_count": cmp["disagreement_count"],
            "nli_negated_count": cmp["nli_negated_count"],
            "scope_negated_count": cmp["scope_negated_count"],
            "scope_speculative_count": cmp["scope_speculative_count"],
            "scope_excluded_count": cmp["scope_excluded_count"],
            "rows": cmp["rows"],
            "nli_negated_examples": cmp["nli_negated_examples"],
            "scope_negated_examples": cmp["scope_negated_examples"],
            "scope_speculative_examples": cmp["scope_speculative_examples"],
            "scope_excluded_examples": cmp["scope_excluded_examples"],
            "disagreements": cmp["disagreements"],
            "sample_disagreements": cmp["disagreements"][:10],
        })

    total_entities = sum(c["entity_count"] for c in report_chunks)
    total_agree = sum(c["agreement_count"] for c in report_chunks)
    total_disagree = sum(c["disagreement_count"] for c in report_chunks)
    total_nli_neg = sum(c["nli_negated_count"] for c in report_chunks)
    total_scope_neg = sum(c["scope_negated_count"] for c in report_chunks)
    total_scope_spec = sum(c["scope_speculative_count"] for c in report_chunks)
    total_scope_excl = sum(c.get("scope_excluded_count", 0) for c in report_chunks)

    print(f"\nPMC: {pmc_id}")
    print(f"Source: {source}")
    print(f"Chunks compared: {len(report_chunks)}")
    print(f"Entities compared: {total_entities}")
    print(f"NLI negated: {total_nli_neg}")
    print(f"Scope negated: {total_scope_neg}")
    print(f"Scope speculative: {total_scope_spec}")
    print(f"Scope excluded: {total_scope_excl}")
    pct = (100 * total_agree / total_entities) if total_entities else 0.0
    print(f"Agreement: {total_agree}/{total_entities} ({pct:.1f}%)")
    print(f"Disagreements: {total_disagree}")

    all_nli_neg = [r for c in report_chunks for r in c.get("nli_negated_examples", [])]
    all_scope_neg = [r for c in report_chunks for r in c.get("scope_negated_examples", [])]
    all_scope_spec = [r for c in report_chunks for r in c.get("scope_speculative_examples", [])]
    all_scope_excl = [r for c in report_chunks for r in c.get("scope_excluded_examples", [])]
    all_disagreements = [r for c in report_chunks for r in c.get("disagreements", [])]

    if all_nli_neg:
        print("\nNLI negated examples:")
        for row in all_nli_neg[:5]:
            _print_example("NLI", row)

    if all_scope_neg:
        print("\nScope negated examples:")
        for row in all_scope_neg[:5]:
            _print_example("Scope", row)

    if all_scope_spec:
        print("\nScope speculative examples:")
        for row in all_scope_spec[:5]:
            _print_example("Speculative", row)

    if all_scope_excl:
        print("\nScope excluded examples:")
        for row in all_scope_excl[:5]:
            _print_example("Excluded", row)

    if all_disagreements:
        print("\nDisagreements:")
        for row in all_disagreements[:5]:
            _print_example("Disagree", row)

    out_dir = Path(__file__).resolve().parents[1] / "negation_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{pmc_id}_{ts}.json"
    md_path = out_dir / f"{pmc_id}_{ts}.md"
    report = {
        "pmc_id": pmc_id,
        "timestamp": ts,
        "source": source,
        "chunks_compared": len(report_chunks),
        "entity_count": total_entities,
        "agreement_count": total_agree,
        "disagreement_count": total_disagree,
        "nli_negated_count": total_nli_neg,
        "scope_negated_count": total_scope_neg,
        "scope_speculative_count": total_scope_spec,
        "scope_excluded_count": total_scope_excl,
        "chunks": report_chunks,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(md_path, report)
    print(f"\nSaved report to {out_path}")
    print(f"Saved readable report to {md_path}")


if __name__ == "__main__":
    main()
