#!/usr/bin/env python3
"""
Analyze FinalTest NER/RE prediction JSON files.

Examples:
    python analyze_predictions.py --pred predictions_test.json
    python analyze_predictions.py --pred predictions_test.json --gold HW_test_data.json
    python analyze_predictions.py --pred predictions_base.json --gold HW_combined_train_data.json --match-by-text
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rel_label(label):
    mapping = {
        "CONTAINS": "CON",
        "USES": "USE",
        "HAS": "HAS",
        "AFFECTS": "AFF",
        "OCCURS_IN": "OCI",
        "LOCATED_IN": "LOI",
    }
    return mapping.get(label, label)


def entity_key(ent):
    return (
        int(ent["start"]),
        int(ent["end"]),
        ent.get("text", ""),
        ent.get("label", ""),
    )


def relation_key(rel):
    return (
        int(rel["head_start"]),
        int(rel["head_end"]),
        rel.get("head", ""),
        rel.get("head_type", ""),
        int(rel["tail_start"]),
        int(rel["tail_end"]),
        rel.get("tail", ""),
        rel.get("tail_type", ""),
        normalize_rel_label(rel.get("label", "")),
    )


def prf(pred_sets, gold_sets):
    tp = fp = fn = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    score = 0.5 * f1 + 0.25 * p + 0.25 * r
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1, "score": score}


def align_gold(pred_data, gold_data, match_by_text):
    if not match_by_text:
        if len(pred_data) != len(gold_data):
            raise ValueError(
                f"Length mismatch: pred={len(pred_data)}, gold={len(gold_data)}. "
                "Use --match-by-text if predictions are a subset."
            )
        return gold_data

    by_text = {item["text"]: item for item in gold_data}
    aligned = []
    missing = []
    for item in pred_data:
        gold = by_text.get(item["text"])
        if gold is None:
            missing.append(item["text"][:80])
            gold = {"text": item["text"], "entities": [], "relations": []}
        aligned.append(gold)
    if missing:
        print(f"WARNING: {len(missing)} prediction texts were not found in gold.")
        for text in missing[:5]:
            print("  missing:", text)
    return aligned


def count_duplicate_spans(items):
    duplicate_entity_items = 0
    duplicate_relation_items = 0
    for item in items:
        ents = [entity_key(e) for e in item.get("entities", [])]
        rels = [relation_key(r) for r in item.get("relations", [])]
        if len(ents) != len(set(ents)):
            duplicate_entity_items += 1
        if len(rels) != len(set(rels)):
            duplicate_relation_items += 1
    return duplicate_entity_items, duplicate_relation_items


def invalid_span_count(items):
    bad_entities = 0
    bad_relations = 0
    for item in items:
        text = item.get("text", "")
        n = len(text)
        for ent in item.get("entities", []):
            start, end = int(ent["start"]), int(ent["end"])
            if start < 0 or end > n or start >= end or text[start:end] != ent.get("text", ""):
                bad_entities += 1
        for rel in item.get("relations", []):
            for prefix in ("head", "tail"):
                start, end = int(rel[f"{prefix}_start"]), int(rel[f"{prefix}_end"])
                if start < 0 or end > n or start >= end or text[start:end] != rel.get(prefix, ""):
                    bad_relations += 1
    return bad_entities, bad_relations


def summarize(name, data):
    ent_counts = Counter()
    rel_counts = Counter()
    ents_per_item = []
    rels_per_item = []
    for item in data:
        ents = item.get("entities", [])
        rels = item.get("relations", [])
        ents_per_item.append(len(ents))
        rels_per_item.append(len(rels))
        ent_counts.update(e.get("label", "") for e in ents)
        rel_counts.update(normalize_rel_label(r.get("label", "")) for r in rels)

    bad_ent, bad_rel = invalid_span_count(data)
    dup_ent_items, dup_rel_items = count_duplicate_spans(data)

    print(f"\n=== {name} ===")
    print(f"texts: {len(data)}")
    print(f"entities: {sum(ents_per_item)} | avg/text: {sum(ents_per_item) / max(len(data), 1):.2f}")
    print(f"relations: {sum(rels_per_item)} | avg/text: {sum(rels_per_item) / max(len(data), 1):.2f}")
    print(f"entity labels: {dict(ent_counts.most_common())}")
    print(f"relation labels: {dict(rel_counts.most_common())}")
    print(f"bad entity spans: {bad_ent}")
    print(f"bad relation spans: {bad_rel}")
    print(f"texts with duplicate entities: {dup_ent_items}")
    print(f"texts with duplicate relations: {dup_rel_items}")


def show_dense_examples(data, limit):
    ranked = sorted(
        enumerate(data),
        key=lambda x: (len(x[1].get("relations", [])), len(x[1].get("entities", []))),
        reverse=True,
    )
    print(f"\n=== Top {limit} relation-heavy examples ===")
    for idx, item in ranked[:limit]:
        print(f"\n#{idx}: entities={len(item.get('entities', []))}, relations={len(item.get('relations', []))}")
        print(item.get("text", "")[:240].replace("\n", " "))
        for rel in item.get("relations", [])[:5]:
            print(
                f"  {rel.get('label')} | {rel.get('head')} ({rel.get('head_type')})"
                f" -> {rel.get('tail')} ({rel.get('tail_type')})"
            )


def compare(pred, gold):
    pred_ent_sets = [set(entity_key(e) for e in item.get("entities", [])) for item in pred]
    gold_ent_sets = [set(entity_key(e) for e in item.get("entities", [])) for item in gold]
    pred_rel_sets = [set(relation_key(r) for r in item.get("relations", [])) for item in pred]
    gold_rel_sets = [set(relation_key(r) for r in item.get("relations", [])) for item in gold]

    ner = prf(pred_ent_sets, gold_ent_sets)
    re = prf(pred_rel_sets, gold_rel_sets)
    total = 0.4 * ner["score"] + 0.6 * re["score"]

    print("\n=== Metrics vs Gold ===")
    print(
        f"NER: P={ner['p']:.4f} R={ner['r']:.4f} F1={ner['f1']:.4f} "
        f"Score={ner['score']:.4f} | TP={ner['tp']} FP={ner['fp']} FN={ner['fn']}"
    )
    print(
        f"RE : P={re['p']:.4f} R={re['r']:.4f} F1={re['f1']:.4f} "
        f"Score={re['score']:.4f} | TP={re['tp']} FP={re['fp']} FN={re['fn']}"
    )
    print(f"Total Score = 0.4*NER + 0.6*RE = {total:.4f}")

    per_label = defaultdict(lambda: [0, 0, 0])
    labels = sorted({k[-1] for s in pred_rel_sets + gold_rel_sets for k in s})
    for label in labels:
        for pred_set, gold_set in zip(pred_rel_sets, gold_rel_sets):
            p = {x for x in pred_set if x[-1] == label}
            g = {x for x in gold_set if x[-1] == label}
            per_label[label][0] += len(p & g)
            per_label[label][1] += len(p - g)
            per_label[label][2] += len(g - p)

    print("\nRE by label:")
    for label, (tp, fp, fn) in sorted(per_label.items()):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {label:4s} P={p:.3f} R={r:.3f} F1={f1:.3f} TP={tp:3d} FP={fp:3d} FN={fn:3d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction JSON path")
    parser.add_argument("--gold", default=None, help="Optional gold JSON path")
    parser.add_argument("--match-by-text", action="store_true", help="Align gold by exact text")
    parser.add_argument("--examples", type=int, default=5, help="How many dense examples to print")
    args = parser.parse_args()

    pred = load_json(args.pred)
    summarize(f"PRED {args.pred}", pred)

    if args.gold:
        gold_raw = load_json(args.gold)
        gold = align_gold(pred, gold_raw, args.match_by_text)
        summarize(f"GOLD {args.gold}", gold)
        compare(pred, gold)

    show_dense_examples(pred, args.examples)


if __name__ == "__main__":
    main()
