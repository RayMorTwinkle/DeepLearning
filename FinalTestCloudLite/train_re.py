"""
FinalTest - RE 训练脚本
    用法:
    python train_re.py --data HW_train_data.json --model microsoft/deberta-v3-large --epochs 12
    python train_re.py --data HW_train_data.json --model ./deberta-v3-large --resume_from ./output_re/best_re_model.pt --epochs 8
    python train_re.py --data HW_train_data.json --model bert-base-uncased --epochs 3  # 本地快速验证

    预期输出: output_re/best_re_model.pt
"""
import os
import sys
import time
import argparse
import shutil
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


def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="RE Training")
    parser.add_argument("--data", type=str, default="HW_train_data.json")
    parser.add_argument("--val_data", type=str, default=None, help="独立验证集路径；提供后不再从训练集切分验证集")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--save_dir", type=str, default="./output_re")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--neg_ratio", type=int, default=3, help="负样本比例")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None, help="从已有 best_re_model.pt 继续训练")
    parser.add_argument("--resume_optimizer", action="store_true", help="同时恢复 optimizer 状态（默认不恢复，更稳）")
    return parser.parse_args()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip, scaler, epoch_idx, total_epochs):
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"RE Train E{epoch_idx}/{total_epochs}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label_id"].to(device)

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
        if (batch_idx + 1) % 20 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / (batch_idx + 1):.3f}",
            })
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_batches


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

    from sklearn.metrics import precision_recall_fscore_support
    positive_labels = list(range(1, NUM_REL_LABELS))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=positive_labels, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=positive_labels, average="macro", zero_division=0
    )

    metrics = {
        "precision": precision, "recall": recall, "f1": f1,
        "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
    }
    metrics["score"] = compute_score(precision, recall, f1)

    return total_loss / len(dataloader), metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ============================================================
    # 0. 环境检查
    # ============================================================
    device = get_device()
    log("=" * 60)
    log(" RE Training Pipeline")
    log("=" * 60)
    log(f"Device:      {device}")
    log(f"Model:       {args.model}")
    log(f"Data:        {args.data}")
    log(f"Save dir:    {args.save_dir}")
    log(f"Max length:  {args.max_length}")
    log(f"Batch size:  {args.batch_size}")
    log(f"Epochs:      {args.epochs}")
    log(f"LR:          {args.lr}")
    log(f"Neg ratio:   {args.neg_ratio}")
    log(f"Dropout:     {args.dropout}")
    log(f"Resume from: {args.resume_from or 'None'}")
    log("-" * 60)

    os.makedirs(args.save_dir, exist_ok=True)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    log("[1/4] Loading data...")
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

    # 统计关系分布
    rel_counts = {}
    for item in data:
        for r in item["relations"]:
            label = r["label"]
            rel_counts[label] = rel_counts.get(label, 0) + 1
    log(f"  Relation distribution:")
    for label, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        log(f"    {label:12s}: {count:4d}")

    # ============================================================
    # 2. 准备 RE 样本
    # ============================================================
    log("[2/4] Preparing RE samples...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer = add_entity_marker_tokens(tokenizer)
    log(f"  Tokenizer: {args.model} (vocab: {tokenizer.vocab_size} + 4 markers)")

    train_samples = prepare_re_samples(
        train_data, tokenizer, max_length=args.max_length,
        neg_ratio=args.neg_ratio, seed=args.seed,
    )
    val_samples = prepare_re_samples(
        val_data, tokenizer, max_length=args.max_length,
        neg_ratio=args.neg_ratio, seed=args.seed,
    )

    # 统计正负样本
    train_pos = sum(1 for s in train_samples if s["label_id"] != REL_LABEL2ID["NONE"])
    train_neg = len(train_samples) - train_pos
    val_pos = sum(1 for s in val_samples if s["label_id"] != REL_LABEL2ID["NONE"])
    val_neg = len(val_samples) - val_pos

    log(f"  Train samples: {len(train_samples)} (pos: {train_pos}, neg: {train_neg})")
    log(f"  Val samples:   {len(val_samples)} (pos: {val_pos}, neg: {val_neg})")
    log(f"  Train batches: {len(train_samples) // args.batch_size + 1}")
    log(f"  Val batches:   {len(val_samples) // args.batch_size + 1}")

    train_dataset = REDataset(train_samples)
    val_dataset = REDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ============================================================
    # 3. 构建模型
    # ============================================================
    log("[3/4] Building model...")
    model = REModel(
        model_name=args.model,
        num_labels=NUM_REL_LABELS,
        dropout=args.dropout,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    # 确保模型用 float32 训练，避免 FP16 溢出
    model = model.float()
    model.to(device)

    start_epoch = 0
    best_score = -1.0
    best_epoch = 0
    best_metrics = None
    resume_checkpoint = None
    if args.resume_from:
        log(f"  Resuming model weights from: {args.resume_from}")
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_metrics = resume_checkpoint.get("val_metrics")
        if best_metrics:
            best_score = float(best_metrics.get("score", -1.0))
            best_epoch = start_epoch
        if args.epochs <= start_epoch:
            raise ValueError(
                f"--epochs ({args.epochs}) must be greater than resumed epoch ({start_epoch}). "
                f"Use --epochs {start_epoch + 1} or larger."
            )
        log(f"  Resumed epoch: {start_epoch}")
        log(f"  Existing best score: {best_score:.4f}")

        # If saving to a different directory, keep a copy of the current best there.
        target_ckpt = os.path.abspath(os.path.join(args.save_dir, "best_re_model.pt"))
        source_ckpt = os.path.abspath(args.resume_from)
        if target_ckpt != source_ckpt:
            shutil.copy2(source_ckpt, target_ckpt)
            tokenizer.save_pretrained(args.save_dir)
            log(f"  Copied current best checkpoint to: {target_ckpt}")

        if not args.resume_optimizer:
            resume_checkpoint.pop("optimizer_state_dict", None)

    total, trainable = count_params(model)
    log(f"  RE labels: {NUM_REL_LABELS} ({REL_ID2LABEL})")
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

    if args.resume_from and args.resume_optimizer and resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        log("  Optimizer state restored.")

    remaining_epochs = args.epochs - start_epoch
    total_steps = len(train_loader) * remaining_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )
    log(f"  Remaining epochs: {remaining_epochs}")
    log(f"  Total remaining steps: {total_steps}")

    # 混合精度训练（AMP）：~2x 加速 + ~40% 省显存
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log(f"  Mixed Precision: {'ON ✅' if use_amp else 'OFF'}")

    # ============================================================
    # 4. 训练
    # ============================================================
    log(f"[4/4] Training (epoch {start_epoch + 1} to {args.epochs})...")
    log("-" * 60)
    patience_counter = 0
    history = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip, scaler, epoch, args.epochs)
        val_loss, val_metrics = validate_re(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        log(f"Epoch {epoch:2d}/{args.epochs} | Time: {epoch_time:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        log(f"  RE → P(pos-micro): {val_metrics['precision']:.4f} | "
            f"R(pos-micro): {val_metrics['recall']:.4f} | "
            f"F1(pos-micro): {val_metrics['f1']:.4f} | "
            f"Score: {val_metrics['score']:.4f}")
        log(f"  RE → P(pos-macro): {val_metrics['precision_macro']:.4f} | "
            f"R(pos-macro): {val_metrics['recall_macro']:.4f} | "
            f"F1(pos-macro): {val_metrics['f1_macro']:.4f}")

        history.append(val_metrics)

        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, os.path.join(args.save_dir, "best_re_model.pt"))
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
    log(" RE Training Complete!")
    log(f" Best Score: {best_score:.4f} at epoch {best_epoch}")
    best_m = best_metrics
    log(f" Best Positive Micro - P: {best_m['precision']:.4f}, R: {best_m['recall']:.4f}, F1: {best_m['f1']:.4f}")
    log(f" Best Positive Macro - P: {best_m['precision_macro']:.4f}, R: {best_m['recall_macro']:.4f}, F1: {best_m['f1_macro']:.4f}")
    log(f" Model saved to: {os.path.abspath(args.save_dir)}/best_re_model.pt")
    log("=" * 60)


if __name__ == "__main__":
    main()
