"""
FinalTest - NER 训练脚本
    python train_ner.py --data ../FinalTest/HW_train_data.json --model bert-base-uncased
    python train_ner.py --data ../FinalTest/HW_train_data.json --model microsoft/deberta-v3-large --epochs 10
"""
import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# 确保能找到同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, split_data, NERDataset,
    NER_LABEL2ID, NER_ID2LABEL, NUM_NER_LABELS, ENTITY_TYPES,
    compute_ner_metrics, compute_score,
    tokens_to_entities_with_offsets,
)
from ner_model import NERModel, get_device


def ner_collate_fn(batch):
    """自定义 collate：处理 tensor 和 string 混合字段"""
    import torch
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch])
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
        "text": texts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="NER Training")
    parser.add_argument("--data", type=str, default="../FinalTest/HW_train_data.json", help="训练数据路径")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="预训练模型名")
    parser.add_argument("--save_dir", type=str, default="./output_ner", help="模型保存目录")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping 轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="NER Train")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_ner(model, dataloader, tokenizer, device):
    """验证并计算 NER 指标"""
    model.eval()
    total_loss = 0.0
    all_pred_entities = []
    all_gold_entities = []

    for batch in tqdm(dataloader, desc="NER Val"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        offset_mappings = batch["offset_mapping"]
        texts = batch["text"]

        # 计算 loss
        outputs = model(input_ids, attention_mask, labels)
        total_loss += outputs["loss"].item()

        # 解码预测
        batch_predictions = model.decode(input_ids, attention_mask)

        for i in range(len(input_ids)):
            seq_len = attention_mask[i].sum().item()
            pred_ids = batch_predictions[i][:seq_len]
            gold_ids = labels[i][:seq_len].cpu().tolist()
            offsets = offset_mappings[i][:seq_len]
            text = texts[i]

            pred_entities = tokens_to_entities_with_offsets(
                input_ids[i][:seq_len].cpu(), pred_ids, offsets, text
            )
            gold_entities = tokens_to_entities_with_offsets(
                input_ids[i][:seq_len].cpu(), gold_ids, offsets, text
            )

            all_pred_entities.append(pred_entities)
            all_gold_entities.append(gold_entities)

    metrics = compute_ner_metrics(all_pred_entities, all_gold_entities)
    score = compute_score(metrics["precision"], metrics["recall"], metrics["f1"])
    metrics["score"] = score

    return total_loss / len(dataloader), metrics


def build_gold_entities_from_data(data_item):
    """从原始数据构造标注实体列表（带 start/end/label 用于评估）"""
    return [
        {"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"]}
        for e in data_item["entities"]
    ]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    print("Loading data...")
    data = load_data(args.data)
    train_data, val_data = split_data(data, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # 加载 tokenizer 和模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = NERModel(
        model_name=args.model,
        num_labels=NUM_NER_LABELS,
        dropout=args.dropout,
    )
    model.to(device)

    # 构建数据集
    train_dataset = NERDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = NERDataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ner_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ner_collate_fn)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )

    # 训练
    best_score = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== NER Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss, val_metrics = validate_ner(model, val_loader, tokenizer, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"NER - P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | Score: {val_metrics['score']:.4f}")

        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_ner_model.pt"))
            tokenizer.save_pretrained(args.save_dir)
            print(f"  ✓ Best model saved (Score: {best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n=== NER Training Complete ===")
    print(f"Best Score: {best_score:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
