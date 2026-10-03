#!/usr/bin/env python3
"""
CNN for CIFAR-10 classification.

Features:
- 4 convolutional layers with batch normalization
- Auto device selection: MPS (Apple Silicon) -> CUDA -> CPU
- Saves best model by validation accuracy
- Comprehensive evaluation metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_classification_report(cm: torch.Tensor, class_names: list[str]) -> str:
    lines: list[str] = []
    header = f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
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

        class_name = class_names[cls] if cls < len(class_names) else str(cls)
        lines.append(f"{class_name:>10} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

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
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {accuracy:>10.4f} {total:>10d}")
    lines.append(
        f"{'macro avg':>10} {macro_precision:>10.4f} {macro_recall:>10.4f} {macro_f1:>10.4f} {total:>10d}"
    )
    lines.append(
        f"{'weighted':>10} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {total:>10d}"
    )
    return "\n".join(lines)


def get_device() -> torch.device:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


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
def evaluate(model: nn.Module, device: torch.device, loader: DataLoader, num_classes: int = 10) -> tuple[float, torch.Tensor]:
    model.eval()
    total_correct = 0
    total_count = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

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
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./best_cnn_cifar10.pth")
    return parser.parse_args()


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main() -> None:
    args = parse_args()

    device = get_device()
    print(f"\nusing device: {device}\n")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    data_dir = Path(args.data_dir)
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"train samples: {len(train_set)} | test samples: {len(test_set)}")

    model = CNN(num_classes=10).to(device)
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
    print(build_classification_report(cm, CIFAR10_CLASS_NAMES))


if __name__ == "__main__":
    main()