"""
FinalTest - 联合推理脚本
    用法:
    python predict.py --ner_model_dir ./output_ner --re_model_dir ./output_re --data HW_train_data.json --output predictions.json --evaluate
    python predict.py --ner_model_dir ./output_ner --re_model_dir ./output_re --test_file test.json --output predictions.json

    预期输出: predictions.json (包含 entities 和 relations)
"""
import os
import sys
import time
import json
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, split_data,
    NER_LABEL2ID, NER_ID2LABEL, NUM_NER_LABELS,
    REL_LABEL2ID, REL_ID2LABEL, NUM_REL_LABELS,
    insert_entity_markers, add_entity_marker_tokens,
    compute_ner_metrics, compute_re_metrics, compute_total_score,
    tokens_to_entities_with_offsets,
)
from ner_model import NERModel, get_device
from re_model import REModel
from transformers import AutoTokenizer


def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_checkpoint_cpu(path):
    """Load a training checkpoint for inference without materializing optimizer tensors eagerly."""
    # 【讲解重点】预测只需要模型权重，不需要训练时 AdamW 优化器的大状态；先在 CPU 读入更省显存。
    kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        checkpoint = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        checkpoint = torch.load(path, **kwargs)

    # Training checkpoints include AdamW states, which are huge and not needed for inference.
    checkpoint.pop("optimizer_state_dict", None)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Joint NER+RE Inference")
    parser.add_argument("--ner_model_dir", type=str, required=True, help="NER 模型目录")
    parser.add_argument("--re_model_dir", type=str, required=True, help="RE 模型目录")
    parser.add_argument("--data", type=str, default=None, help="训练数据（有标注，用于验证评估）")
    parser.add_argument("--test_file", type=str, default=None, help="测试文件（纯文本或带标注）")
    parser.add_argument("--output", type=str, default="predictions.json", help="预测输出文件")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--re_threshold", type=float, default=0.9, help="RE 置信度阈值")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--re_max_length", type=int, default=384)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="有标注数据时进行评估")
    parser.add_argument(
        "--use_test_entities",
        action="store_true",
        help="test_file 已提供 entities 时，直接使用这些实体，只预测 relations",
    )
    return parser.parse_args()


def load_ner_model(model_dir, device):
    """加载 NER 模型"""
    # 【讲解重点】根据 best_ner_model.pt 里的训练参数重建 NERModel，再塞回保存好的最优权重。
    log(f"  Loading NER checkpoint from: {model_dir}")
    checkpoint = load_checkpoint_cpu(os.path.join(model_dir, "best_ner_model.pt"))
    model_name = checkpoint["args"]["model"]
    log(f"  NER model: {model_name} (epoch {checkpoint['epoch']})")
    log(f"  NER val score: {checkpoint['val_metrics']['score']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = NERModel(
        model_name=model_name,
        num_labels=NUM_NER_LABELS,
        dropout=checkpoint["args"].get("dropout", 0.1),
        init_from_config=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, tokenizer


def load_re_model(model_dir, device):
    """加载 RE 模型"""
    # 【讲解重点】RE tokenizer 需要重新加入实体标记 token，模型词表大小也要对应扩展。
    log(f"  Loading RE checkpoint from: {model_dir}")
    checkpoint = load_checkpoint_cpu(os.path.join(model_dir, "best_re_model.pt"))
    model_name = checkpoint["args"]["model"]
    log(f"  RE model: {model_name} (epoch {checkpoint['epoch']})")
    log(f"  RE val score: {checkpoint['val_metrics']['score']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer = add_entity_marker_tokens(tokenizer)
    model = REModel(
        model_name=model_name,
        num_labels=NUM_REL_LABELS,
        dropout=checkpoint["args"].get("dropout", 0.1),
        init_from_config=True,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, tokenizer


@torch.no_grad()
def predict_ner(model, tokenizer, texts, device, max_length=384, batch_size=12):
    """对文本列表进行 NER 预测"""
    # 【讲解重点】NER 推理：文本 -> tokenizer offset -> 模型 BIOES 标签 -> 还原成字符级实体 span。
    all_entities = []
    for i in tqdm(range(0, len(texts), batch_size), desc="NER Inference"):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts, max_length=max_length, truncation=True,
            padding=True, return_offsets_mapping=True, return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        offset_mappings = encodings["offset_mapping"]

        batch_preds = model.decode(input_ids, attention_mask)

        for j, (text, pred_ids, offsets, mask) in enumerate(
            zip(batch_texts, batch_preds, offset_mappings, attention_mask)
        ):
            seq_len = mask.sum().item()
            entities = tokens_to_entities_with_offsets(
                input_ids[j][:seq_len].cpu(), pred_ids[:seq_len],
                offsets[:seq_len].tolist(), text,
            )
            all_entities.append(entities)
    return all_entities


@torch.no_grad()
def predict_re_on_entities(model, tokenizer, texts, all_entities, device,
                           max_length=384, threshold=0.9, batch_size=32):
    """在预测的实体上进行 RE 推理"""
    # 【讲解重点】RE 推理：对每篇文本的实体两两组合，逐对分类；置信度低于 threshold 的关系会被过滤。
    all_relations = []
    total_pairs = 0
    total_preds = 0

    for text, entities in tqdm(zip(texts, all_entities), total=len(texts), desc="RE Inference"):
        relations = []
        n = len(entities)
        if n < 2:
            all_relations.append(relations)
            continue

        pair_data = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                marked = insert_entity_markers(text, entities[i], entities[j])
                pair_data.append((entities[i], entities[j], marked))

        total_pairs += len(pair_data)

        for k in range(0, len(pair_data), batch_size):
            batch = pair_data[k:k + batch_size]
            encodings = tokenizer(
                [p[2] for p in batch], max_length=max_length,
                truncation=True, padding=True, return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            pred_labels, probs = model.predict(input_ids, attention_mask)

            for m, (e1, e2, _) in enumerate(batch):
                label_id = pred_labels[m].item()
                prob = probs[m][label_id].item()
                if label_id != 0 and prob >= threshold:
                    total_preds += 1
                    relations.append({
                        "head": e1["text"], "head_start": e1["start"],
                        "head_end": e1["end"], "head_type": e1["label"],
                        "tail": e2["text"], "tail_start": e2["start"],
                        "tail_end": e2["end"], "tail_type": e2["label"],
                        "label": REL_ID2LABEL[label_id],
                    })
        all_relations.append(relations)

    log(f"  Total entity pairs: {total_pairs}, predicted relations: {total_preds}")
    return all_relations


def main():
    args = parse_args()
    device = get_device()
    log("=" * 60)
    log(" Joint NER+RE Inference")
    log("=" * 60)
    log(f"Device:       {device}")
    log(f"NER model:    {args.ner_model_dir}")
    log(f"RE model:     {args.re_model_dir}")
    log(f"Threshold:    {args.re_threshold}")
    log("-" * 60)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    log("[1/4] Loading data...")
    if args.data:
        log(f"  Source: {args.data}")
        data = load_data(args.data)
        _, val_data = split_data(data, val_ratio=args.val_ratio, seed=args.seed)
        texts = [item["text"] for item in val_data]
        gold_entities = [item["entities"] for item in val_data]
        gold_relations = [item["relations"] for item in val_data]
        is_labeled = True
    elif args.test_file:
        log(f"  Source: {args.test_file}")
        with open(args.test_file, "r") as f:
            test_data = json.load(f)
        if isinstance(test_data[0], dict) and "text" in test_data[0]:
            texts = [item["text"] for item in test_data]
            gold_entities = [item.get("entities", []) for item in test_data]
            gold_relations = [item.get("relations", []) for item in test_data]
            is_labeled = any(len(e) > 0 for e in gold_entities)
        else:
            texts = test_data
            gold_entities = [[]] * len(texts)
            gold_relations = [[]] * len(texts)
            is_labeled = False
    else:
        raise ValueError("Must provide --data or --test_file")

    log(f"  Texts to process: {len(texts)}")
    log(f"  Labeled: {is_labeled}")

    has_provided_entities = any(len(e) > 0 for e in gold_entities)
    use_provided_entities = args.test_file and args.use_test_entities and has_provided_entities
    # 【讲解重点】如果测试文件已经给了 entities，--use_test_entities 会跳过 NER，只用 RE 预测 relations。

    # ============================================================
    # 2. 加载模型
    # ============================================================
    log("[2/4] Loading models...")
    if use_provided_entities:
        log("  Skipping NER model because --use_test_entities is enabled.")
        ner_model = None
        ner_tokenizer = None
    else:
        ner_model, ner_tokenizer = load_ner_model(args.ner_model_dir, device)
    re_model, re_tokenizer = load_re_model(args.re_model_dir, device)

    # ============================================================
    # 3. NER 推理
    # ============================================================
    if use_provided_entities:
        log("[3/4] Using entities provided by test_file...")
        pred_entities = gold_entities
        total_pred_ent = sum(len(e) for e in pred_entities)
        log(f"  Provided entities: {total_pred_ent}")
    else:
        log("[3/4] Running NER inference...")
        t0 = time.time()
        pred_entities = predict_ner(
            ner_model, ner_tokenizer, texts, device,
            max_length=args.max_length, batch_size=args.batch_size,
        )
        total_pred_ent = sum(len(e) for e in pred_entities)
        log(f"  Done in {time.time() - t0:.1f}s")
        log(f"  Predicted entities: {total_pred_ent}")

    # ============================================================
    # 4. RE 推理
    # ============================================================
    log("[4/4] Running RE inference...")
    t0 = time.time()
    pred_relations = predict_re_on_entities(
        re_model, re_tokenizer, texts, pred_entities, device,
        max_length=args.re_max_length, threshold=args.re_threshold,
    )
    total_pred_rel = sum(len(r) for r in pred_relations)
    log(f"  Done in {time.time() - t0:.1f}s")
    log(f"  Predicted relations: {total_pred_rel}")

    # ============================================================
    # 5. 保存预测
    # ============================================================
    log("Saving predictions...")
    # 【讲解重点】最终交付文件格式：每条文本对应 entities 和 relations 两个预测列表。
    predictions = []
    for text, entities, relations in zip(texts, pred_entities, pred_relations):
        predictions.append({"text": text, "entities": entities, "relations": relations})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    log(f"  Output: {os.path.abspath(args.output)}")

    # ============================================================
    # 6. 评估（如果有标注）
    # ============================================================
    if is_labeled and args.evaluate:
        # 【讲解重点】只有输入文件自带标准答案并且加了 --evaluate，才会在本地计算验证分数。
        log("Evaluating...")
        label_map = {
            "CONTAINS": "CON", "USES": "USE", "HAS": "HAS",
            "AFFECTS": "AFF", "OCCURS_IN": "OCI", "LOCATED_IN": "LOI",
        }
        gold_relations_norm = []
        for item_rels in gold_relations:
            norm = []
            for rel in item_rels:
                r = dict(rel)
                r["label"] = label_map.get(r["label"], r["label"])
                norm.append(r)
            gold_relations_norm.append(norm)

        ner_metrics = compute_ner_metrics(pred_entities, gold_entities)
        re_metrics = compute_re_metrics(pred_relations, gold_relations_norm)
        total_score = compute_total_score(ner_metrics, re_metrics)

        log("=" * 60)
        log(" Evaluation Results")
        log("=" * 60)
        ner_score = 0.5 * ner_metrics['f1'] + 0.25 * ner_metrics['precision'] + 0.25 * ner_metrics['recall']
        re_score = 0.5 * re_metrics['f1'] + 0.25 * re_metrics['precision'] + 0.25 * re_metrics['recall']
        log(f"  NER  → P: {ner_metrics['precision']:.4f}  R: {ner_metrics['recall']:.4f}  F1: {ner_metrics['f1']:.4f}  Score: {ner_score:.4f}")
        log(f"  RE   → P: {re_metrics['precision']:.4f}  R: {re_metrics['recall']:.4f}  F1: {re_metrics['f1']:.4f}  Score: {re_score:.4f}")
        log(f"  Total Score = 0.4×NER + 0.6×RE = {total_score:.4f}")
        log(f"  Entities:  predicted {total_pred_ent} / gold {sum(len(e) for e in gold_entities)}")
        log(f"  Relations: predicted {total_pred_rel} / gold {sum(len(r) for r in gold_relations_norm)}")
        log("=" * 60)


if __name__ == "__main__":
    main()
