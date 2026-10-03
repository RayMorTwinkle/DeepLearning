"""
FinalTest - NER 训练脚本
    用法:
    python train_ner.py --data HW_train_data.json --model ./deberta-v3-large
    python train_ner.py --data HW_train_data.json --model bert-base-uncased --epochs 3  # 本地快速验证

    预期输出: output_ner/best_ner_model.pt
"""
import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_data, split_data, NERDataset,
    NER_LABEL2ID, NER_ID2LABEL, NUM_NER_LABELS, ENTITY_TYPES,
    compute_ner_metrics, compute_score,
    tokens_to_entities_with_offsets,
)
from ner_model import NERModel, get_device


def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ner_collate_fn(batch):
    """自定义 collate：处理 tensor 和 string 混合字段"""
    # 【讲解重点】DataLoader 默认只能拼 tensor。
    # 我们额外保留 text / offset_mapping / gold_entities，方便验证时把 token 预测还原成实体。
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    offset_mapping = torch.stack([item["offset_mapping"] for item in batch])
    texts = [item["text"] for item in batch]
    gold_entities = [item["gold_entities"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "offset_mapping": offset_mapping,
        "text": texts,
        "gold_entities": gold_entities,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="NER Training")
    parser.add_argument("--data", type=str, default="HW_train_data.json", help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None, help="独立验证集路径；提供后不再从训练集切分验证集")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large", help="预训练模型名")
    parser.add_argument("--save_dir", type=str, default="./output_ner", help="模型保存目录")
    parser.add_argument("--max_length", type=int, default=384, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=12, help="批次大小")
    parser.add_argument("--epochs", type=int, default=8, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping 轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def count_params(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip, scaler, epoch_idx, total_epochs):
    # 【讲解重点】NER 单轮训练循环：
    # 前向计算 loss -> 反向传播 -> 梯度裁剪 -> optimizer 更新 -> 学习率调度。
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"NER Train E{epoch_idx}/{total_epochs}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # 混合精度 autocast
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_was_run = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer_was_run = True

        if optimizer_was_run:
            scheduler.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / (batch_idx + 1):.2f}",
            })
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_batches


@torch.no_grad()
def validate_ner(model, dataloader, device):
    """验证并计算 NER 指标"""
    # 【讲解重点】验证阶段不只看 loss。
    # 还会把 token 级预测还原为实体 span，再和 gold_entities 计算 P/R/F1。
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
        gold_batch_entities = batch["gold_entities"]

        outputs = model(input_ids, attention_mask, labels)
        total_loss += outputs["loss"].item()

        batch_predictions = model.decode(input_ids, attention_mask)

        for i in range(len(input_ids)):
            seq_len = attention_mask[i].sum().item()
            pred_ids = batch_predictions[i][:seq_len]
            offsets = offset_mappings[i][:seq_len]
            text = texts[i]

            pred_entities = tokens_to_entities_with_offsets(
                input_ids[i][:seq_len].cpu(), pred_ids, offsets, text
            )

            all_pred_entities.append(pred_entities)
            all_gold_entities.append(gold_batch_entities[i])

    metrics = compute_ner_metrics(all_pred_entities, all_gold_entities)
    score = compute_score(metrics["precision"], metrics["recall"], metrics["f1"])
    metrics["score"] = score

    return total_loss / len(dataloader), metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ============================================================
    # 0. 环境检查
    # ============================================================
    device = get_device()
    log("=" * 60)
    log(" NER Training Pipeline")
    log("=" * 60)
    log(f"Device:     {device}")
    log(f"Model:      {args.model}")
    log(f"Data:       {args.data}")
    log(f"Save dir:   {args.save_dir}")
    log(f"Max length: {args.max_length}")
    log(f"Batch size: {args.batch_size}")
    log(f"Epochs:     {args.epochs}")
    log(f"LR:         {args.lr}")
    log(f"Dropout:    {args.dropout}")
    log("-" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    log("[1/4] Loading data...")
    # 【讲解重点】训练数据入口。
    # Notebook 里传入的是合并后的 HW_combined_train_data.json。
    data = load_data(args.data)
    if args.val_data:
        train_data = data
        val_data = load_data(args.val_data)
        log(f"  Train source: {args.data}")
        log(f"  Val source:   {args.val_data}")
    else:
        train_data, val_data = split_data(data, val_ratio=args.val_ratio, seed=args.seed)
    log(f"  Total train samples: {len(data)}")
    log(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # 统计实体分布
    entity_counts = {}
    for item in data:
        for e in item["entities"]:
            entity_counts[e["label"]] = entity_counts.get(e["label"], 0) + 1
    log(f"  Entity distribution:")
    for label in ENTITY_TYPES:
        count = entity_counts.get(label, 0)
        log(f"    {label:8s}: {count:4d}")

    # ============================================================
    # 2. 构建数据集
    # ============================================================
    log("[2/4] Building datasets...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    log(f"  Tokenizer: {args.model} (vocab size: {tokenizer.vocab_size})")

    # 【讲解重点】这里触发 data_utils.py 里的 NER 数据处理：
    # 原始 JSON -> BIOES 标签 -> tokenizer 对齐 -> tensor dataset。
    train_dataset = NERDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = NERDataset(val_data, tokenizer, max_length=args.max_length)
    log(f"  Train batches: {len(train_dataset) // args.batch_size + 1}")
    log(f"  Val batches:   {len(val_dataset) // args.batch_size + 1}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ner_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ner_collate_fn)

    # ============================================================
    # 3. 构建模型
    # ============================================================
    log("[3/4] Building model...")
    # 【讲解重点】构建 NER 模型：DeBERTa-v3-base + Linear + CRF。
    model = NERModel(
        model_name=args.model,
        num_labels=NUM_NER_LABELS,
        dropout=args.dropout,
    )
    # 确保模型用 float32 训练，避免 FP16 溢出
    model = model.float()
    model.to(device)

    total, trainable = count_params(model)
    log(f"  NER labels: {NUM_NER_LABELS}")
    log(f"  Total params:    {total:,}")
    log(f"  Trainable params: {trainable:,}")

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
    log(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")

    # 混合精度训练（AMP）：~2x 加速 + ~40% 省显存
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log(f"  Mixed Precision: {'ON ✅' if use_amp else 'OFF'}")
    log(f"  Effective batch size: {args.batch_size} (×{'2' if use_amp else '1'} speed)")

    # ============================================================
    # 4. 训练
    # ============================================================
    log(f"[4/4] Training ({args.epochs} epochs)...")
    log("-" * 60)
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip, scaler, epoch, args.epochs)
        val_loss, val_metrics = validate_ner(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        log(f"Epoch {epoch:2d}/{args.epochs} | Time: {epoch_time:.0f}s | "
            f"Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")
        log(f"  NER → P: {val_metrics['precision']:.4f} | "
            f"R: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Score: {val_metrics['score']:.4f}")

        history.append(val_metrics)

        # 【讲解重点】只保存验证集 score 最高的模型。
        # 这样即使后面过拟合，最终推理仍使用最好的一轮。
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
            }, os.path.join(args.save_dir, "best_ner_model.pt"))
            tokenizer.save_pretrained(args.save_dir)
            log(f"  ✓ New best model! (Score: {best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"  ⚠ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ============================================================
    # 5. 总结
    # ============================================================
    log("=" * 60)
    log(" NER Training Complete!")
    log(f" Best Score: {best_score:.4f} at epoch {best_epoch}")
    log(f" Best P: {history[best_epoch-1]['precision']:.4f}")
    log(f" Best R: {history[best_epoch-1]['recall']:.4f}")
    log(f" Best F1: {history[best_epoch-1]['f1']:.4f}")
    log(f" Model saved to: {os.path.abspath(args.save_dir)}/best_ner_model.pt")
    log("=" * 60)


if __name__ == "__main__":
    main()
