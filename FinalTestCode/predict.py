"""
FinalTest - 联合推理脚本
    用法:
    python predict.py --ner_model ./output_ner --re_model ./output_re --data ../FinalTest/HW_train_data.json --output predictions.json
    python predict.py --ner_model ./output_ner --re_model ./output_re --test_file test.json --output predictions.json
"""
import os
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description="Joint NER+RE Inference")
    parser.add_argument("--ner_model_dir", type=str, required=True, help="NER 模型目录")
    parser.add_argument("--re_model_dir", type=str, required=True, help="RE 模型目录")
    parser.add_argument("--data", type=str, default=None, help="训练数据（有标注，用于验证评估）")
    parser.add_argument("--test_file", type=str, default=None, help="测试文件（纯文本或带标注）")
    parser.add_argument("--output", type=str, default="predictions.json", help="预测输出文件")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--re_threshold", type=float, default=0.5, help="RE 置信度阈值")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--re_max_length", type=int, default=384)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="有标注数据时进行评估")
    return parser.parse_args()


def load_ner_model(model_dir, device):
    """加载 NER 模型"""
    checkpoint = torch.load(
        os.path.join(model_dir, "best_ner_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model_name = checkpoint["args"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = NERModel(
        model_name=model_name,
        num_labels=NUM_NER_LABELS,
        dropout=checkpoint["args"].get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer


def load_re_model(model_dir, device):
    """加载 RE 模型"""
    checkpoint = torch.load(
        os.path.join(model_dir, "best_re_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model_name = checkpoint["args"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer = add_entity_marker_tokens(tokenizer)
    model = REModel(
        model_name=model_name,
        num_labels=NUM_REL_LABELS,
        dropout=checkpoint["args"].get("dropout", 0.1),
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_ner(model, tokenizer, texts, device, max_length=256, batch_size=16):
    """对文本列表进行 NER 预测，返回实体列表（带字符偏移）"""
    all_entities = []

    for i in tqdm(range(0, len(texts), batch_size), desc="NER Inference"):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        offset_mappings = encodings["offset_mapping"]

        # Viterbi 解码
        batch_preds = model.decode(input_ids, attention_mask)

        for j, (text, pred_ids, offsets, mask) in enumerate(
            zip(batch_texts, batch_preds, offset_mappings, attention_mask)
        ):
            seq_len = mask.sum().item()
            entities = tokens_to_entities_with_offsets(
                input_ids[j][:seq_len].cpu(),
                pred_ids[:seq_len],
                offsets[:seq_len].tolist(),
                text,
            )
            all_entities.append(entities)

    return all_entities


@torch.no_grad()
def predict_re_on_entities(
    model, tokenizer, texts, all_entities, device,
    max_length=384, threshold=0.5, batch_size=32,
):
    """
    在预测的实体上进行 RE 推理
    对所有实体对进行关系分类

    Returns:
        all_relations: 每条文本的关系列表
    """
    all_relations = []

    for text, entities in tqdm(zip(texts, all_entities), total=len(texts), desc="RE Inference"):
        relations = []
        n = len(entities)

        if n < 2:
            all_relations.append(relations)
            continue

        # 生成所有实体对
        pair_data = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # 构造带标记的文本
                marked = insert_entity_markers(text, entities[i], entities[j])
                pair_data.append((entities[i], entities[j], marked))

        # 分批推理
        for k in range(0, len(pair_data), batch_size):
            batch = pair_data[k:k + batch_size]
            batch_texts = [p[2] for p in batch]

            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            pred_labels, probs = model.predict(input_ids, attention_mask)

            for m, (e1, e2, _) in enumerate(batch):
                label_id = pred_labels[m].item()
                prob = probs[m][label_id].item()

                if label_id != 0 and prob >= threshold:
                    label_name = REL_ID2LABEL[label_id]
                    relations.append({
                        "head": e1["text"],
                        "head_start": e1["start"],
                        "head_end": e1["end"],
                        "head_type": e1["label"],
                        "tail": e2["text"],
                        "tail_start": e2["start"],
                        "tail_end": e2["end"],
                        "tail_type": e2["label"],
                        "label": label_name,
                    })

        all_relations.append(relations)

    return all_relations


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # 加载模型
    print("Loading NER model...")
    ner_model, ner_tokenizer = load_ner_model(args.ner_model_dir, device)

    print("Loading RE model...")
    re_model, re_tokenizer = load_re_model(args.re_model_dir, device)

    # 加载数据
    if args.data:
        print(f"Loading data from: {args.data}")
        data = load_data(args.data)
        _, val_data = split_data(data, val_ratio=args.val_ratio, seed=args.seed)
        texts = [item["text"] for item in val_data]
        gold_entities = [item["entities"] for item in val_data]
        gold_relations = [item["relations"] for item in val_data]
        data_items = val_data
        is_labeled = True
    elif args.test_file:
        print(f"Loading test file: {args.test_file}")
        with open(args.test_file, "r") as f:
            test_data = json.load(f)
        if isinstance(test_data, list):
            # 可能带标注也可能不带
            if isinstance(test_data[0], dict) and "text" in test_data[0]:
                texts = [item["text"] for item in test_data]
                gold_entities = [item.get("entities", []) for item in test_data]
                gold_relations = [item.get("relations", []) for item in test_data]
                is_labeled = any(len(e) > 0 for e in gold_entities)
                data_items = test_data
            else:
                texts = test_data
                gold_entities = [[]] * len(texts)
                gold_relations = [[]] * len(texts)
                is_labeled = False
                data_items = [{"text": t} for t in texts]
        else:
            raise ValueError("Test file must be a JSON list")
    else:
        raise ValueError("Must provide --data or --test_file")

    print(f"Texts to process: {len(texts)}")

    # Step 1: NER 推理
    print("\n[Step 1] Running NER inference...")
    pred_entities = predict_ner(
        ner_model, ner_tokenizer, texts, device,
        max_length=args.max_length, batch_size=args.batch_size,
    )

    # Step 2: RE 推理
    print("\n[Step 2] Running RE inference...")
    pred_relations = predict_re_on_entities(
        re_model, re_tokenizer, texts, pred_entities, device,
        max_length=args.re_max_length, threshold=args.re_threshold,
    )

    # Step 3: 组合并保存预测结果
    print("\n[Step 3] Saving predictions...")
    predictions = []
    for i, (text, entities, relations) in enumerate(zip(texts, pred_entities, pred_relations)):
        pred_item = {
            "text": text,
            "entities": entities,
            "relations": relations,
        }
        predictions.append(pred_item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to: {args.output}")

    # Step 4: 评估（如果有标注）
    if is_labeled and args.evaluate:
        print("\n[Step 4] Evaluating...")

        # 统一 gold 关系标签
        label_map = {
            "CONTAINS": "CON", "USES": "USE", "HAS": "HAS",
            "AFFECTS": "AFF", "OCCURS_IN": "OCI", "LOCATED_IN": "LOI",
        }
        gold_relations_normalized = []
        for item_rels in gold_relations:
            normalized = []
            for rel in item_rels:
                r = dict(rel)
                r["label"] = label_map.get(r["label"], r["label"])
                normalized.append(r)
            gold_relations_normalized.append(normalized)

        ner_metrics = compute_ner_metrics(pred_entities, gold_entities)
        re_metrics = compute_re_metrics(pred_relations, gold_relations_normalized)
        total_score = compute_total_score(ner_metrics, re_metrics)

        print(f"\n{'='*60}")
        print(f"NER - P: {ner_metrics['precision']:.4f} | "
              f"R: {ner_metrics['recall']:.4f} | F1: {ner_metrics['f1']:.4f}")
        print(f"RE  - P: {re_metrics['precision']:.4f} | "
              f"R: {re_metrics['recall']:.4f} | F1: {re_metrics['f1']:.4f}")
        print(f"Total Score: {total_score:.4f}")
        print(f"{'='*60}")

        # 统计信息
        total_pred_ent = sum(len(e) for e in pred_entities)
        total_pred_rel = sum(len(r) for r in pred_relations)
        total_gold_ent = sum(len(e) for e in gold_entities)
        total_gold_rel = sum(len(r) for r in gold_relations_normalized)
        print(f"\nEntities: predicted {total_pred_ent} / gold {total_gold_ent}")
        print(f"Relations: predicted {total_pred_rel} / gold {total_gold_rel}")


if __name__ == "__main__":
    main()
