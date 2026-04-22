#!/usr/bin/env python3
"""Interactive visual dashboard for MNIST 3-layer MLP training."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def bi(zh: str, en: str) -> str:
    return f"{zh} / {en}"


def get_device() -> torch.device:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ThreeLayerMLP(nn.Module):
    """3 layers total: input -> hidden -> output."""

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


@st.cache_data(show_spinner=False)
def load_mnist(data_dir: str) -> tuple[datasets.MNIST, datasets.MNIST]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    root = Path(data_dir)
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train_set, test_set


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1)

            total_loss += loss.item()
            total_correct += (pred == target).sum().item()
            total_count += target.size(0)

            all_preds.append(pred.cpu())
            all_labels.append(target.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total_count, 1)
    return avg_loss, acc, torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def plot_network(hidden_dim: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title("网络结构图 / Network Topology (3 Layers Total)")
    ax.axis("off")

    x_positions = [0.1, 0.5, 0.9]
    layer_names = ["Input (784)", f"Hidden ({hidden_dim})", "Output (10)"]
    visible_nodes = [10, 10, 10]
    colors = ["#6baed6", "#74c476", "#fd8d3c"]

    for x, name, n, color in zip(x_positions, layer_names, visible_nodes, colors):
        ys = np.linspace(0.1, 0.9, n)
        for y in ys:
            circle = plt.Circle((x, y), 0.018, color=color, ec="black", lw=0.4)
            ax.add_patch(circle)
        ax.text(x, 0.98, name, ha="center", va="top", fontsize=10, weight="bold")

    y_left = np.linspace(0.1, 0.9, visible_nodes[0])
    y_mid = np.linspace(0.1, 0.9, visible_nodes[1])
    y_right = np.linspace(0.1, 0.9, visible_nodes[2])
    for yl in y_left:
        for ym in y_mid:
            ax.plot([x_positions[0], x_positions[1]], [yl, ym], color="gray", alpha=0.15, lw=0.5)
    for ym in y_mid:
        for yr in y_right:
            ax.plot([x_positions[1], x_positions[2]], [ym, yr], color="gray", alpha=0.15, lw=0.5)

    ax.text(
        0.5,
        0.02,
        "说明: 图中节点数量做了压缩展示 / Note: nodes are visually compressed for readability",
        ha="center",
        fontsize=9,
    )
    return fig


def plot_sample_images(dataset: datasets.MNIST, n: int = 12) -> plt.Figure:
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes = axes.flatten()
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(0), cmap="gray")
        axes[i].set_title(f"标签 label={label}")
        axes[i].axis("off")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵（测试集）/ Confusion Matrix (Test Set)")
    ax.set_xlabel("预测类别 / Predicted")
    ax.set_ylabel("真实类别 / True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="MNIST 可视化训练面板 | Visual Trainer", layout="wide")
    st.title("MNIST 三层 MLP 可视化训练面板 / 3-Layer MLP Visual Trainer")
    st.caption("输入层 -> 隐藏层 -> 输出层 | Input -> Hidden -> Output | 实时训练可视化 / Realtime visualization")

    with st.sidebar:
        st.header("超参数设置 / Hyperparameters")
        preset = st.selectbox(
            "预设方案 / Preset",
            ["平衡 Balanced", "快速 Fast", "正则化 Regularized"],
            help="快速选择一组常用默认参数 / Pick a starter configuration.",
        )
        if preset == "平衡 Balanced":
            default_hidden, default_dropout, default_lr, default_wd = 256, 0.10, 8e-4, 1e-4
        elif preset == "快速 Fast":
            default_hidden, default_dropout, default_lr, default_wd = 192, 0.05, 1e-3, 1e-5
        else:
            default_hidden, default_dropout, default_lr, default_wd = 320, 0.20, 6e-4, 3e-4

        hidden_dim = st.slider(
            "隐藏层维度 hidden_dim",
            64,
            1024,
            default_hidden,
            step=32,
            help="隐藏层神经元数量。越大容量越强，但可能更慢、也更容易过拟合。",
        )
        dropout = st.slider(
            "Dropout 比例 dropout",
            0.0,
            0.6,
            float(default_dropout),
            step=0.05,
            help="训练时随机丢弃神经元比例。增大可抑制过拟合。",
        )
        lr = st.number_input(
            "学习率 learning_rate",
            min_value=1e-5,
            max_value=1e-1,
            value=float(default_lr),
            format="%.5f",
            help="每次参数更新步长。过大易震荡，过小收敛慢。",
        )
        weight_decay = st.number_input(
            "权重衰减 weight_decay",
            min_value=0.0,
            max_value=1e-2,
            value=float(default_wd),
            format="%.5f",
            help="L2 正则项，帮助泛化，防止参数过大。",
        )
        batch_size = st.select_slider(
            "批大小 batch_size",
            options=[32, 64, 128, 256, 512],
            value=128,
            help="每个 batch 的样本数量。大 batch 更稳，小 batch 更省显存。",
        )
        epochs = st.slider(
            "训练轮数 epochs",
            1,
            30,
            10,
            help="完整遍历训练集的次数。",
        )
        max_batches = st.slider(
            "每轮最多 batch（演示加速）max batches/epoch",
            50,
            1000,
            1000,
            step=50,
            help="为了演示速度可以限制每轮训练的 batch 数。",
        )
        data_dir = st.text_input(
            "数据目录 data_dir",
            value="./data",
            help="MNIST 下载和读取目录。",
        )
        save_path = st.text_input(
            "模型保存路径 save_path",
            value="./best_mlp_mnist.pth",
            help="验证集准确率最佳时保存模型参数。",
        )
        optimizer_name = st.selectbox(
            "优化器 optimizer",
            ["AdamW", "SGD"],
            help="AdamW 通常收敛更快，SGD 更传统。",
        )

    device = get_device()
    st.success(f"当前设备 / Current device: `{device}`")

    train_set, test_set = load_mnist(data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("数据样本 / Dataset Samples")
        st.caption("展示 MNIST 原始输入图像与标签，帮助确认数据读取是否正确。")
        st.pyplot(plot_sample_images(train_set), clear_figure=True)
    with col_b:
        st.subheader("网络结构可视化 / Network Visualization")
        st.caption("当前模型为三层总结构：输入层 -> 隐藏层 -> 输出层。")
        st.pyplot(plot_network(hidden_dim), clear_figure=True)

    model = ThreeLayerMLP(hidden_dim=hidden_dim, dropout=dropout).to(device)
    model_param_count = count_params(model)
    st.info(f"可训练参数量 / Trainable parameters: `{model_param_count:,}`")

    if optimizer_name == "AdamW":
        optimizer: optim.Optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    with st.expander("参数与图像说明 / Notes for Parameters & Plots", expanded=False):
        st.markdown(
            "- `train_loss`：训练集平均损失，越低越好。\n"
            "- `val_loss`：验证集平均损失，用于观察泛化。\n"
            "- `train_acc`：训练集准确率。\n"
            "- `val_acc`：验证集准确率，建议优先关注。\n"
            "- 混淆矩阵：行是真实类别，列是预测类别，对角线越亮越好。"
        )

    st.markdown("### 训练控制 / Training Controls")
    start = st.button("开始训练 Start Training", type="primary")

    if not start:
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    metric_box = st.empty()
    loss_chart_title = st.empty()
    loss_chart_box = st.empty()
    acc_chart_title = st.empty()
    acc_chart_box = st.empty()
    cm_box = st.empty()
    cm_title = st.empty()
    epoch_table = st.empty()

    history: list[dict[str, float]] = []
    best_acc = 0.0
    save_path_obj = Path(save_path)

    total_steps = max(epochs * min(len(train_loader), max_batches), 1)
    current_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            train_loss_sum += loss.item()
            train_correct += (pred == target).sum().item()
            train_count += target.size(0)

            current_step += 1
            progress.progress(min(current_step / total_steps, 1.0))
            if batch_idx % 20 == 0:
                status.write(
                    f"轮次 Epoch {epoch}/{epochs} | 批次 Batch {batch_idx} | "
                    f"损失 Loss {loss.item():.4f} | 当前训练准确率 Running Train Acc {100.0 * train_correct / max(train_count,1):.2f}%"
                )

        train_loss = train_loss_sum / max(min(len(train_loader), max_batches), 1)
        train_acc = 100.0 * train_correct / max(train_count, 1)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, test_loader, device, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path_obj)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "best_val_acc": best_acc,
            }
        )
        df_hist = pd.DataFrame(history)

        metric_box.metric("最佳验证准确率 / Best Validation Accuracy", f"{best_acc:.2f}%")

        loss_chart_title.markdown("#### Loss 曲线 / Loss Curves")
        loss_chart_box.line_chart(df_hist.set_index("epoch")[["train_loss", "val_loss"]])

        acc_chart_title.markdown("#### 准确率曲线 / Accuracy Curves")
        acc_chart_box.line_chart(df_hist.set_index("epoch")[["train_acc", "val_acc"]])

        show_df = df_hist.rename(
            columns={
                "epoch": "epoch",
                "train_loss": "train_loss",
                "val_loss": "val_loss",
                "train_acc": "train_acc(%)",
                "val_acc": "val_acc(%)",
                "best_val_acc": "best_val_acc(%)",
            }
        )
        epoch_table.dataframe(show_df, use_container_width=True)

        cm = confusion_matrix(val_labels, val_preds)
        cm_title.markdown("#### 混淆矩阵 / Confusion Matrix")
        cm_box.pyplot(plot_confusion_matrix(cm), clear_figure=True)
        time.sleep(0.05)

    status.success(f"训练完成 / Training finished. 最佳验证准确率 Best validation accuracy: {best_acc:.2f}%")
    st.write(f"最佳模型保存路径 / Best model path: `{save_path_obj.resolve()}`")


if __name__ == "__main__":
    main()
