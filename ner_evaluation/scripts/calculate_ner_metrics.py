#!/usr/bin/env python3
"""
scripts/calculate_ner_metrics.py
─────────────────────────────────
Calculate precision, recall, and F1 for NER comparison results.

This script scores the two model outputs against an ensemble-consensus proxy
derived from the entities both models agree on.

It reports the same comparison lenses described in the README:

1. Net entity yield
2. Category-specific coverage shift
3. Consensus precision proxy
4. Span precision / boundary quality

Usage:
    python scripts/calculate_ner_metrics.py data/ner_comparison/PMC8160999_*.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def calculate_metrics(predicted: Set[str], reference: Set[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 from two normalized string sets."""
    if not predicted or not reference:
        return 0.0, 0.0, 0.0

    true_positives = len(predicted & reference)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calculate_partial_match_metrics(
    predicted: List[Dict],
    reference: List[Dict],
    match_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calculate metrics allowing partial token overlap.

    Two entities match if:
    1. They have at least `match_threshold` token overlap.
    2. They have the same label.
    """

    def normalize_text(text: str) -> str:
        return text.lower().strip()

    def texts_overlap(text1: str, text2: str) -> float:
        tokens1 = set(normalize_text(text1).split())
        tokens2 = set(normalize_text(text2).split())
        if not tokens1 or not tokens2:
            return 0.0
        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return overlap / union if union > 0 else 0.0

    matched_pred: set[int] = set()
    matched_ref: set[int] = set()

    for i, pred_ent in enumerate(predicted):
        pred_text = pred_ent.get("text", "")
        pred_label = pred_ent.get("label", "")

        best_match_idx = None
        best_overlap = 0.0

        for j, ref_ent in enumerate(reference):
            if j in matched_ref:
                continue

            if pred_label != ref_ent.get("label", ""):
                continue

            overlap = texts_overlap(pred_text, ref_ent.get("text", ""))
            if overlap >= match_threshold and overlap > best_overlap:
                best_overlap = overlap
                best_match_idx = j

        if best_match_idx is not None:
            matched_pred.add(i)
            matched_ref.add(best_match_idx)

    precision = len(matched_pred) / len(predicted) if predicted else 0.0
    recall = len(matched_ref) / len(reference) if reference else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_against_ensemble(data: Dict) -> Dict:
    """Evaluate the two model outputs using the ensemble-consensus proxy."""
    scispacy_ents = data.get("scispacy", [])
    hunflair_ents = data.get("hunflair", [])

    scispacy_set = {e["text"].lower().strip(): e for e in scispacy_ents if e.get("text")}
    hunflair_set = {e["text"].lower().strip(): e for e in hunflair_ents if e.get("text")}

    sci_keys = set(scispacy_set.keys())
    hun_keys = set(hunflair_set.keys())
    consensus_keys = sci_keys & hun_keys
    sci_exclusive = sci_keys - hun_keys
    hun_exclusive = hun_keys - sci_keys
    total_unique = sci_keys | hun_keys

    avg_len_sci = sum(len(k) for k in sci_keys) / len(sci_keys) if sci_keys else 0.0
    avg_len_hun = sum(len(k) for k in hun_keys) / len(hun_keys) if hun_keys else 0.0

    return {
        "total_unique_entities": len(total_unique),
        "consensus_count": len(consensus_keys),
        "scispacy": {
            "total_count": len(sci_keys),
            "exclusive_count": len(sci_exclusive),
            "avg_span_len": round(avg_len_sci, 1),
            "overlap_ratio": round(len(consensus_keys) / len(sci_keys), 3) if sci_keys else 0.0,
        },
        "hunflair": {
            "total_count": len(hun_keys),
            "exclusive_count": len(hun_exclusive),
            "avg_span_len": round(avg_len_hun, 1),
            "overlap_ratio": round(len(consensus_keys) / len(hun_keys), 3) if hun_keys else 0.0,
        },
        "exclusive_gains": {
            "scispacy_only": sorted(sci_exclusive)[:20],
            "hunflair_only": sorted(hun_exclusive)[:20],
        },
        "comparison_lenses": {
            "net_entity_yield": {
                "scispacy_gain": len(sci_exclusive),
                "hunflair_gain": len(hun_exclusive),
                "total_unique_entities": len(total_unique),
            },
            "category_specific_coverage_shift": {
                "scispacy_total": len(sci_keys),
                "hunflair_total": len(hun_keys),
                "consensus_count": len(consensus_keys),
            },
            "high_confidence_consensus_precision_proxy": {
                "true_positive_proxy": len(consensus_keys),
                "scispacy_precision_proxy": round(len(consensus_keys) / len(sci_keys), 3) if sci_keys else 0.0,
                "hunflair_precision_proxy": round(len(consensus_keys) / len(hun_keys), 3) if hun_keys else 0.0,
            },
            "span_precision_and_boundary_quality": {
                "scispacy_avg_span_len": round(avg_len_sci, 1),
                "hunflair_avg_span_len": round(avg_len_hun, 1),
            },
        },
    }


def evaluate_per_label(data: Dict) -> Dict:
    """Calculate precision, recall, and F1 broken down by entity type."""
    scispacy_ents = data.get("scispacy", [])
    hunflair_ents = data.get("hunflair", [])

    scispacy_by_label = defaultdict(set)
    hunflair_by_label = defaultdict(set)

    for ent in scispacy_ents:
        text = ent.get("text")
        if text:
            scispacy_by_label[ent.get("label", "UNKNOWN")].add(text.lower().strip())

    for ent in hunflair_ents:
        text = ent.get("text")
        if text:
            hunflair_by_label[ent.get("label", "UNKNOWN")].add(text.lower().strip())

    results = {}
    for label in sorted(set(scispacy_by_label) | set(hunflair_by_label)):
        sci_set = scispacy_by_label.get(label, set())
        hun_set = hunflair_by_label.get(label, set())
        consensus = sci_set & hun_set
        if not consensus:
            continue

        label_results = {}
        for model_name, ent_set in (("scispacy", sci_set), ("hunflair", hun_set)):
            if not ent_set:
                continue
            prec, rec, f1 = calculate_metrics(ent_set, consensus)
            label_results[model_name] = {
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "count": len(ent_set),
            }

        results[label] = {
            "consensus_count": len(consensus),
            "models": label_results,
        }

    return results


def process_comparison_file(filepath: Path) -> Dict:
    """Load and evaluate a single NER comparison file."""
    print(f"Processing: {filepath.name}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return {
        "pmc_id": data.get("pmc_id", filepath.stem.split("_")[0]),
        "input_file": str(filepath),
        "timestamp": data.get("timestamp", ""),
        "chunks_processed": data.get("chunks", 0),
        "evaluation_method": "ensemble_consensus_2plus",
        "overall_metrics": evaluate_against_ensemble(data),
        "per_label_metrics": evaluate_per_label(data),
    }


def print_report(report: Dict) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 80)
    print(f"NER EVALUATION REPORT: {report['pmc_id']}")
    print("=" * 80)

    overall = report["overall_metrics"]
    lenses = overall.get("comparison_lenses", {})

    print(f"\nTotal Unique Entities (Union across models): {overall['total_unique_entities']}")
    print(f"Consensus Agreement (Found by BOTH models)   : {overall['consensus_count']}")

    if lenses:
        print("\n" + "-" * 80)
        print("COMPARISON LENSES")
        print("-" * 80)
        net = lenses.get("net_entity_yield", {})
        cat = lenses.get("category_specific_coverage_shift", {})
        conf = lenses.get("high_confidence_consensus_precision_proxy", {})
        span = lenses.get("span_precision_and_boundary_quality", {})
        print(
            f"Net Entity Yield       : scispacy +{net.get('scispacy_gain', 0)}, "
            f"hunflair +{net.get('hunflair_gain', 0)}"
        )
        print(
            f"Category Coverage      : scispacy {cat.get('scispacy_total', 0)} "
            f"vs hunflair {cat.get('hunflair_total', 0)}"
        )
        print(
            f"Consensus Precision    : scispacy {conf.get('scispacy_precision_proxy', 0.0):.1%}, "
            f"hunflair {conf.get('hunflair_precision_proxy', 0.0):.1%}"
        )
        print(
            f"Span Quality           : scispacy {span.get('scispacy_avg_span_len', 0.0)} chars, "
            f"hunflair {span.get('hunflair_avg_span_len', 0.0)} chars"
        )

    print("\n" + "-" * 80)
    print("MODEL COMPARISON (Net Yield & Span Quality)")
    print("-" * 80)
    print(f"{'Model':<15} {'Total Entities':>15} {'Exclusive Gain':>16} {'Overlap Ratio':>15} {'Avg Span Len':>14}")
    print("-" * 80)

    for model_name in ("scispacy", "hunflair"):
        metrics = overall.get(model_name, {})
        print(
            f"{model_name:<15} "
            f"{metrics.get('total_count', 0):>15} "
            f"{'+' + str(metrics.get('exclusive_count', 0)):>16} "
            f"{metrics.get('overlap_ratio', 0.0):>14.1%} "
            f"{str(metrics.get('avg_span_len', 0.0)) + ' chars':>14}"
        )

    print("\n" + "-" * 80)
    print("EXCLUSIVE ENTITY SAMPLES")
    print("-" * 80)
    gains = overall.get("exclusive_gains", {})
    hun_g = gains.get("hunflair_only", [])
    sci_g = gains.get("scispacy_only", [])
    print(f"HunFlair Only ({len(hun_g)} samples):")
    print("  " + ", ".join(f"'{e}'" for e in hun_g[:10]))
    print(f"\nscispaCy Only ({len(sci_g)} samples):")
    print("  " + ", ".join(f"'{e}'" for e in sci_g[:10]))
    print("=" * 80 + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/calculate_ner_metrics.py <comparison_file.json> [...]")
        print("\nExample:")
        print("  python scripts/calculate_ner_metrics.py data/ner_comparison/PMC8160999_*.json")
        sys.exit(1)

    all_reports = []
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if filepath.name.endswith("_metrics.json"):
            continue
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue

        report = process_comparison_file(filepath)
        all_reports.append(report)
        print_report(report)

        output_path = filepath.parent / f"{filepath.stem}_metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Metrics saved to: {output_path}\n")

    if len(all_reports) > 1:
        print("\n" + "=" * 80)
        print("AGGREGATE METRICS ACROSS ALL FILES")
        print("=" * 80)
        for model_name in ("scispacy", "hunflair", "combined"):
            precisions = [r["overall_metrics"].get(model_name, {}).get("precision", 0) for r in all_reports]
            recalls = [r["overall_metrics"].get(model_name, {}).get("recall", 0) for r in all_reports]
            f1s = [r["overall_metrics"].get(model_name, {}).get("f1", 0) for r in all_reports]

            if any(precisions):
                print(f"\n{model_name.upper()}:")
                print(f"  Avg Precision: {sum(precisions) / len(precisions):.1%}")
                print(f"  Avg Recall:    {sum(recalls) / len(recalls):.1%}")
                print(f"  Avg F1:        {sum(f1s) / len(f1s):.1%}")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
