#!/usr/bin/env python3
"""
scripts/compare_ner_pmc.py
──────────────────────────
Fetch a real PMC paper using the PRODUCTION FETCHER and compare what each NER 
model finds. This ensures fair comparison by using the exact same text cleaning
that the production pipeline uses.

Usage:
    # NER server must be running first (for HunFlair):
    #   HUNFLAIR_ENABLED=true ./start-ner-server.sh

    .venv/bin/python scripts/compare_ner_pmc.py PMC8160999
    .venv/bin/python scripts/compare_ner_pmc.py PMC7327471
    .venv/bin/python scripts/compare_ner_pmc.py PMC9500000
    .venv/bin/python scripts/compare_ner_pmc.py PMC11975295
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import datetime
import pathlib
import textwrap
from pathlib import Path
from collections import Counter, defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fetcher.fetcher import Fetcher

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
RED = "\033[91m"

def _h(text, col): return f"{col}{BOLD}{text}{RESET}"
def _d(text):      return f"{DIM}{text}{RESET}"


def _banner(title: str, col: str = BOLD) -> None:
    print(f"\n{col}{'━' * 78}{RESET}")
    print(f"{col}  {title}{RESET}")
    print(f"{col}{'━' * 78}{RESET}")


def _section(title: str, col: str = BOLD) -> None:
    print(f"\n{col}{'═' * 78}{RESET}")
    print(f"{col}  {title}{RESET}")
    print(f"{col}{'═' * 78}{RESET}")


def fetch_pmc_chunks(pmc_id: str, use_coref: bool = True) -> tuple[list[dict], str, bool]:
    """
    Fetch and chunk a PMC paper using the PRODUCTION FETCHER.
    Falls back to data/checkpoints/<pmc_id>/layer3_chunks.json if present.
    """
    proj_root = Path(__file__).parent.parent.parent
    ckpt_file = proj_root / "data" / "checkpoints" / pmc_id / "layer3_chunks.json"
    if ckpt_file.exists():
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if chunks and any(c.get("text", "").strip() for c in chunks):
                full_text = " ".join(c.get("text", "") for c in chunks)
                print(f"  {GREEN}✓{RESET} Loaded {len(chunks)} chunks from checkpoint ({ckpt_file})", flush=True)
                return chunks, full_text, False
        except Exception:
            pass

    num_id = pmc_id.upper().lstrip("PMC")
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={num_id}&rettype=full&retmode=xml"
    )
    paper_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
    source = {"name": "pmc", "type": "xml", "rate_limit_delay": 0.34}
    
    coref_url = os.getenv("COREF_SERVICE_URL", "").strip() if use_coref else ""
    coref_used = False
    
    if coref_url:
        print(f"{_d('Fetching')} {pmc_id} using production fetcher (with coref resolution) …", flush=True)
        try:
            fetcher = Fetcher(coref_url=coref_url)
            if fetcher.coref_client.health_check():
                print(f"  {GREEN}✓{RESET} Coref service online at {coref_url}", flush=True)
                coref_used = True
            else:
                fetcher.coref_client.health_check = lambda: True
        except Exception:
            fetcher = Fetcher(coref_url="http://dummy:9999")
            fetcher.coref_client.health_check = lambda: True
    else:
        print(f"{_d('Fetching')} {pmc_id} using production fetcher …", flush=True)
        fetcher = Fetcher(coref_url="http://dummy:9999")
        fetcher.coref_client.health_check = lambda: True
    
    try:
        chunks = fetcher.fetch(
            url=fetch_url,
            source=source,
            document_id=pmc_id,
            paper_url=paper_url,
            verbose=False
        )
    except Exception:
        chunks = []

    # Fallback if XML fetch produced empty text
    valid_chunks = [c for c in chunks if c.get("text", "").strip()] if chunks else []
    if not valid_chunks:
        print(f"  {YELLOW}⚠ Standard XML returned empty text. Trying NCBI BioC API fallback…{RESET}", flush=True)
        try:
            bioc_url = f"https://www.ncbi.nlm.nih.gov/research/bpubchem/rest/v1/pmc/{pmc_id}/bioc/json"
            req = urllib.request.Request(bioc_url, headers={"User-Agent": "BioSemantic/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                bioc_data = json.loads(resp.read().decode())
                extracted_texts = []
                for doc in bioc_data.get("documents", []):
                    for passg in doc.get("passages", []):
                        txt = passg.get("text", "").strip()
                        if txt:
                            extracted_texts.append(txt)
                if extracted_texts:
                    full_bioc = "\n\n".join(extracted_texts)
                    words = full_bioc.split()
                    chunk_size = 350
                    chunk_list = []
                    for i in range(0, len(words), chunk_size):
                        c_text = " ".join(words[i:i+chunk_size])
                        chunk_list.append({
                            "chunk_index": len(chunk_list),
                            "text": c_text,
                            "section": "body",
                            "document_id": pmc_id,
                            "source_name": "pmc",
                            "source_url": paper_url,
                        })
                    chunks = chunk_list
                    print(f"  {GREEN}✓ BioC Fallback succeeded: {len(chunks)} chunks extracted.{RESET}", flush=True)
        except Exception as fallback_err:
            print(f"  {RED}BioC Fallback failed: {fallback_err}{RESET}", flush=True)

    if not chunks or not any(c.get("text", "").strip() for c in chunks):
        sys.exit(f"[ERROR] Could not extract text for {pmc_id} from XML or BioC fallback.")
    
    full_text = " ".join(c.get("text", "") for c in chunks)
    status = f"{GREEN}with coref{RESET}" if coref_used else f"{_d('without coref')}"
    print(f"  ✓ {len(chunks)} chunks, {len(full_text):,} chars total ({status})", flush=True)
    return chunks, full_text, coref_used

def extract_chunk_texts(chunks: list[dict]) -> list[str]:
    """Extract just the text field from production chunks."""
    return [c["text"] for c in chunks]


def run_spacy_ner(chunks: list[str]) -> list[dict]:
    """Run only the scispaCy ensemble (no server calls)."""
    import spacy

    def _load(key, default):
        name = os.getenv(key, default).strip()
        if not name:
            return None
        try:
            return spacy.load(name)
        except OSError:
            print(f"  {_d(f'[scispaCy] model {name!r} not found — skipping')}", flush=True)
            return None

    models = [m for m in [
        _load("NER_MODEL_1", "en_ner_bc5cdr_md"),
        _load("NER_MODEL_2", "en_ner_jnlpba_md"),
        _load("NER_MODEL_3", "en_ner_bionlp13cg_md"),
        _load("NER_MODEL_4", "en_core_sci_lg"),
    ] if m is not None]

    entities: list[dict] = []
    seen: set[str] = set()

    for chunk_text in chunks:
        span_data: list[tuple[int, int, str]] = []
        base_doc = models[0](chunk_text) if models else None
        if base_doc is None:
            continue
        span_data += [(e.start_char, e.end_char, e.label_) for e in base_doc.ents]
        for nlp in models[1:]:
            doc = nlp(chunk_text)
            span_data += [(e.start_char, e.end_char, e.label_) for e in doc.ents]

        all_spans = []
        for s, e, lbl in span_data:
            span = base_doc.char_span(s, e, label=lbl, alignment_mode="expand")
            if span:
                all_spans.append(span)
        base_doc.ents = spacy.util.filter_spans(all_spans)

        for ent in base_doc.ents:
            key = ent.text.lower().strip()
            if key in seen or len(key) < 2:
                continue
            seen.add(key)
            entities.append({
                "text":   ent.text,
                "label":  ent.label_,
                "source": "scispacy",
            })
    return entities


def run_server_ner(chunks: list[str], endpoint: str, timeout: int = 120) -> list[dict]:
    """
    Call the NER server at `endpoint` (e.g. /ner/hunflair)
    and aggregate results across all chunks.
    HunFlair on CPU can take 60-90s per chunk — use a generous timeout.
    """
    server_url = os.getenv("NER_SERVER_URL", "http://localhost:8001").rstrip("/")
    url = f"{server_url}{endpoint}"
    entities: list[dict] = []
    seen: set[str] = set()

    for i, chunk_text in enumerate(chunks, 1):
        print(f"  {_d(f'chunk {i}/{len(chunks)} …')}", end="\r", flush=True)
        payload = json.dumps({"text": chunk_text}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  {RED}[Server] HTTP {e.code} on chunk {i}: {e.reason}{RESET}", flush=True)
            continue
        except Exception as e:
            print(f"  {RED}[Server] Error on chunk {i}: {e}{RESET}", flush=True)
            continue

        for ent in result.get("entities", []):
            key = (ent.get("text") or "").lower().strip()
            if key in seen or len(key) < 2:
                continue
            seen.add(key)
            entities.append(ent)

    return entities


LABEL_COLORS: dict[str, str] = {
    "DISEASE":               RED,
    "GENE_OR_GENE_PRODUCT":  GREEN,
    "CHEMICAL":              CYAN,
    "ORGANISM":              YELLOW,
    "CELL_LINE": "\033[95m",
    "DRUG":                  CYAN,
    "SYMPTOM":               RED,
    "PROCEDURE":             "\033[94m",   
    "CLINICAL_INTERVENTION": "\033[94m",
    "ANATOMY":               "\033[93m",
    "EVENT":                 "\033[93m",
    "CLINICAL_MEASUREMENT":  "\033[96m",
}

def _label_col(label: str) -> str:
    return LABEL_COLORS.get(label, "")

def _fmt_ent(ent: dict) -> str:
    col   = _label_col(ent["label"])
    text  = ent["text"][:40]
    label = ent["label"]
    src   = ent.get("source", "?")
    conf  = ent.get("confidence", "")
    conf_str = f" {_d(f'({conf:.2f})')}" if isinstance(conf, float) else ""
    return f"  {col}{text:<42}{RESET} {BOLD}{label:<30}{RESET} {_d(src)}{conf_str}"


def print_section(title: str, entities: list[dict], col: str = BOLD):
    by_type = defaultdict(list)
    for e in entities:
        by_type[e["label"]].append(e)

    _banner(f"{title}  ({len(entities)} unique entities)", col)

    if not entities:
        print(f"  {_d('(none)')}")
        return

    for label, ents in sorted(by_type.items(), key=lambda x: -len(x[1])):
        col_l = _label_col(label)
        print(f"\n  {col_l}{BOLD}{label}{RESET}  {_d(f'({len(ents)})')}")
        for ent in sorted(ents, key=lambda e: e["text"].lower()):
            print(_fmt_ent(ent))


def print_comparison(
    spacy_ents: list[dict],
    hun_ents: list[dict],
    hun_ents_set: set[str],
    spacy_set: set[str],
):
    _section("COMPARISON SUMMARY")

    rows = (
        ("scispaCy only", len(spacy_ents), ""),
        ("+ HunFlair", len(hun_ents), f"(+{len(hun_ents_set - spacy_set)} new)"),
        ("Combined (both)", len(spacy_set | hun_ents_set), ""),
    )
    for name, count, note in rows:
        print(f"  {name:<30} {count:>4}  {_d(note)}")

    # Unique contributions
    hun_unique = hun_ents_set - spacy_set

    print(f"\n{BOLD}  Unique to HunFlair          ({len(hun_unique)}){RESET}")
    for t in sorted(hun_unique)[:15]:
        print(f"    {CYAN}+{RESET} {t}")
    if len(hun_unique) > 15:
        print(f"    {_d(f'… and {len(hun_unique)-15} more')}")

    # Entity type breakdown
    print(f"\n{BOLD}  Entity type breakdown:{RESET}")
    all_ents = {**{e["text"].lower(): e for e in spacy_ents},
                **{e["text"].lower(): e for e in hun_ents}}
    by_type = Counter(e["label"] for e in all_ents.values())
    for label, cnt in by_type.most_common():
        col = _label_col(label)
        bar = "█" * min(cnt, 40)
        print(f"    {col}{label:<30}{RESET}  {cnt:>4}  {_d(bar)}")



def main():
    pmc_id = sys.argv[1] if len(sys.argv) > 1 else "PMC8160999"
    use_coref = "--no-coref" not in sys.argv  # Allow disabling coref with --no-coref flag

    server_url = os.getenv("NER_SERVER_URL", "").strip()
    health = {}
    if not server_url:
        print(f"{YELLOW}[!] NER_SERVER_URL is not set. Server-mode NER will be skipped.{RESET}")
        print(f"{YELLOW}    Start the server: HUNFLAIR_ENABLED=true ./start-ner-server.sh{RESET}\n")
    else:
        try:
            health_req = urllib.request.Request(f"{server_url}/health")
            with urllib.request.urlopen(health_req, timeout=5) as r:
                health = json.loads(r.read())
            models_ready = health.get("models_ready", [])
            print(f"{GREEN}NER server online{RESET} — models ready: {models_ready}\n")
        except Exception as e:
            print(f"{RED}[ERROR] NER server not reachable at {server_url}: {e}{RESET}")
            print(f"Start it with:  HUNFLAIR_ENABLED=true ./start-ner-server.sh\n")
            server_url = ""

    production_chunks, full_text, coref_used = fetch_pmc_chunks(pmc_id, use_coref=use_coref)
    
    chunk_texts = extract_chunk_texts(production_chunks)
    
    coref_status = f"{GREEN}WITH{RESET}" if coref_used else f"{YELLOW}WITHOUT{RESET}"
    print(f"  ✓ Using production-cleaned chunks ({coref_status} coreference resolution)\n")

    # Print a short excerpt
    excerpt = " ".join(full_text.split()[:120])
    print(f"{_d('Excerpt:')}")
    for line in textwrap.wrap(excerpt, 78):
        print(f"  {_d(line)}")

    print(f"\n{BOLD}Running scispaCy NER …{RESET}", flush=True)
    spacy_ents = run_spacy_ner(chunk_texts)
    spacy_set  = {e["text"].lower() for e in spacy_ents}
    print_section("scispaCy Only", spacy_ents, BOLD)

    hun_ents: list[dict]    = []
    hun_ents_set: set[str]  = set()
    if server_url and "hunflair" in health.get("models_ready", []):
        print(f"\n{BOLD}Running HunFlair via server …{RESET}", flush=True)
        hun_ents    = run_server_ner(chunk_texts, "/ner/hunflair")
        hun_ents_set = {e["text"].lower() for e in hun_ents}

        new_hun = [e for e in hun_ents if e["text"].lower() not in spacy_set]
        print_section(
            f"HunFlair  — {len(new_hun)} NEW vs scispaCy",
            new_hun, GREEN
        )
    else:
        print(f"\n{_d('[HunFlair] skipped — server not ready or hunflair not loaded')}")

    print_comparison(spacy_ents, hun_ents, hun_ents_set, spacy_set)

    out_dir = pathlib.Path("data/ner_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{pmc_id}_{ts}.json"
    report = {
        "pmc_id":          pmc_id,
        "timestamp":       ts,
        "chunks":          len(chunk_texts),
        "fetcher_version": "production",
        "coref_resolution": coref_used,
        "coref_url":       os.getenv("COREF_SERVICE_URL", "") if coref_used else None,
        "scispacy":        spacy_ents,
        "hunflair":        hun_ents,
        "scispacy_count":  len(spacy_ents),
        "hunflair_count":  len(hun_ents),
        "hunflair_new_count": len(hun_ents_set - spacy_set),
    }
    out_path.write_text(json.dumps(report, indent=2))
    
    coref_note = f" (coref: {GREEN}enabled{RESET})" if coref_used else f" (coref: {YELLOW}disabled{RESET})"
    print(f"\n{GREEN}✓ Report saved → {out_path}{coref_note}{RESET}\n")


if __name__ == "__main__":
    main()
