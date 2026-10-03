"""
FinalTest - RE 训练脚本
    python train_re.py --data ../FinalTest/HW_train_data.json --model bert-base-uncased --ner_model_dir ./output_ner
    python train_re.py --data ../FinalTest/HW_train_data.json --model microsoft/deberta-v3-large --epochs 12
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, split_data, prepare_re_samples, REDataset,
    REL_LABEL2ID, REL_ID2LABEL, NUM_REL_LABELS,
    compute_re_metrics, compute_score, add_entity_marker_tokens,
)
from re_model import REModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="RE Training")
    parser.add_argument("--data", type=str, default="../FinalTest/HW_train_data.json")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--save_dir", type=str, default="./output_re")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--neg_ratio", type=int, default=3, help="负样本比例")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="RE Train")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

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
def validate_re(model, dataloader, device):
    """验证并计算 RE 指标"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="RE Val"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

        outputs = model(input_ids, attention_mask, labels)
        total_loss += outputs["loss"].item()

        preds = torch.argmax(outputs["logits"], dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # 计算微观指标
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    # 排除 NONE 类计算 macro（关注正类）
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }
    metrics["score"] = compute_score(precision, recall, f1)

    return total_loss / len(dataloader), metrics


def build_re_gold_from_data(val_data):
    """从原始数据构建验证用的关系列表"""
    gold_relations = []
    for item in val_data:
        item_rels = []
        for rel in item.get("relations", []):
            # 统一标签名
            label = rel["label"]
            label_map = {
                "CONTAINS": "CON", "USES": "USE", "HAS": "HAS",
                "AFFECTS": "AFF", "OCCURS_IN": "OCI", "LOCATED_IN": "LOI",
            }
            label = label_map.get(label, label)
            item_rels.append({
                "head_start": rel["head_start"],
                "head_end": rel["head_end"],
                "head_type": rel["head_type"],
                "tail_start": rel["tail_start"],
                "tail_end": rel["tail_end"],
                "tail_type": rel["tail_type"],
                "label": label,
            })
        gold_relations.append(item_rels)
    return gold_relations


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

    # 加载 tokenizer 并添加实体标记
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = add_entity_marker_tokens(tokenizer)

    # 准备 RE 样本
    print("Preparing RE training samples...")
    train_samples = prepare_re_samples(
        train_data, tokenizer,
        max_length=args.max_length,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    val_samples = prepare_re_samples(
        val_data, tokenizer,
        max_length=args.max_length,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_dataset = REDataset(train_samples)
    val_dataset = REDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 构建模型
    print("Building RE model...")
    model = REModel(
        model_name=args.model,
        num_labels=NUM_REL_LABELS,
        dropout=args.dropout,
    )
    # 调整 embedding 大小以适应新增的 special tokens
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )

    # 训练
    best_score = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== RE Epoch {epoch}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss, val_metrics = validate_re(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"RE - P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | Score: {val_metrics['score']:.4f}")
        print(f"RE Macro - P: {val_metrics['precision_macro']:.4f} | "
              f"R: {val_metrics['recall_macro']:.4f} | F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_re_model.pt"))
            tokenizer.save_pretrained(args.save_dir)
            print(f"  ✓ Best model saved (Score: {best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n=== RE Training Complete ===")
    print(f"Best Score: {best_score:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
