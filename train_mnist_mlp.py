#!/usr/bin/env python3
"""
MLP for MNIST classification.

Features:
- 3 layers total (input + hidden + output)
- Auto device selection: MPS (Apple Silicon) -> CUDA -> CPU
- Saves best model by validation accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_classification_report(cm: torch.Tensor) -> str:
    lines: list[str] = []
    header = f"{'class':>7} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    total = int(cm.sum().item())
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for cls in range(cm.size(0)):
        tp = float(cm[cls, cls].item())
        row_sum = float(cm[cls, :].sum().item())
        col_sum = float(cm[:, cls].sum().item())
        fp = col_sum - tp
        fn = row_sum - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(row_sum)

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

        lines.append(f"{cls:>7} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

    accuracy = float(torch.diag(cm).sum().item()) / total if total > 0 else 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    num_classes = cm.size(0)
    for cls in range(num_classes):
        tp = float(cm[cls, cls].item())
        row_sum = float(cm[cls, :].sum().item())
        col_sum = float(cm[:, cls].sum().item())
        fp = col_sum - tp
        fn = row_sum - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes

    weighted_precision /= total if total > 0 else 1.0
    weighted_recall /= total if total > 0 else 1.0
    weighted_f1 /= total if total > 0 else 1.0

    lines.append("-" * len(header))
    lines.append(f"{'accuracy':>7} {'':>10} {'':>10} {accuracy:>10.4f} {total:>10d}")
    lines.append(
        f"{'macro avg':>7} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total:>10d}"
    )
    lines.append(
        f"{'weighted':>7} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total:>10d}"
    )
    return "\n".join(lines)


def get_device() -> torch.device:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_count += target.size(0)

        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx:4d}/{len(loader):4d} | loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = 100.0 * total_correct / total_count
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, device: torch.device, loader: DataLoader) -> tuple[float, torch.Tensor]:
    model.eval()
    total_correct = 0
    total_count = 0
    cm = torch.zeros((10, 10), dtype=torch.int64)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        preds = logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_count += target.size(0)
        for t, p in zip(target.cpu(), preds.cpu()):
            cm[t.long(), p.long()] += 1

    return 100.0 * total_correct / total_count, cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./best_mlp_mnist.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()
    print(f"\nusing device: {device}\n")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_dir = Path(args.data_dir)
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"train samples: {len(train_set)} | test samples: {len(test_set)}")

    model = MLP(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = Path(args.save_path)

    print("\nstart training\n" + "=" * 50)
    for epoch in range(1, args.epochs + 1):
        print(f"\nepoch {epoch}/{args.epochs}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        test_acc, _ = evaluate(model, device, test_loader)

        print(
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.2f}% | "
            f"test acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"saved best model to: {save_path} (acc: {best_acc:.2f}%)")

    print("\n" + "=" * 50)
    print(f"done. best test accuracy: {best_acc:.2f}%")
    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    final_acc, cm = evaluate(model, device, test_loader)
    print(f"final test accuracy (best model): {final_acc:.2f}%")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nclassification report:")
    print(build_classification_report(cm))


if __name__ == "__main__":
    main()
