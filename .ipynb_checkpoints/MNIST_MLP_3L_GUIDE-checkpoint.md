# MNIST 三层 MLP 完整学习笔记

这份文档配套代码：
- 脚本版：[train_mnist_mlp.py](/Users/ray/Documents/Fork/DeepLearning/train_mnist_mlp.py)
- Notebook 版：[mnist_mlp_notebook.ipynb](/Users/ray/Documents/Fork/DeepLearning/mnist_mlp_notebook.ipynb)

目标：
- 老师要求“包括输入层和输出层，总共三层”
- 我们实现的是：输入层 -> 隐藏层 -> 输出层

---

## 1. 这份代码到底在做什么

任务是手写数字识别（MNIST）：
- 输入：28x28 灰度图片
- 输出：数字类别 0~9

核心流程：
1. 选设备（MPS/CUDA/CPU）
2. 下载并读取 MNIST
3. 定义三层 MLP
4. 训练（反向传播）
5. 测试（准确率）
6. 保存最佳模型

---

## 2. 模型结构（“三层”怎么数）

我们按“包括输入层和输出层”来数：

1. 输入层：784 维（28x28 展平）
2. 隐藏层：`hidden_dim`（默认 256）
3. 输出层：10 维（10 个类别）

在代码里对应：
- `nn.Flatten()`：把 28x28 变成 784
- `nn.Linear(784, hidden_dim)`：输入层到隐藏层
- `nn.Linear(hidden_dim, 10)`：隐藏层到输出层

中间组件：
- `ReLU`：增加非线性表达能力
- `Dropout`：减少过拟合（训练时随机屏蔽一部分神经元）

---

## 3. 数据集（数据库）是什么、组成是什么

MNIST 的组成：
- 训练集：60,000 张图片
- 测试集：10,000 张图片
- 每张图：28x28，单通道灰度图
- 标签：0~9 的整数

在代码中：
- `datasets.MNIST(..., train=True)`：训练集
- `datasets.MNIST(..., train=False)`：测试集

预处理（`transform`）：
- `ToTensor()`：把像素转成 tensor，范围到 [0,1]
- `Normalize((0.1307,), (0.3081,))`：标准化，训练更稳定

---

## 4. 各模块角色（谁负责什么）

### 4.1 `get_device()`
角色：
- 自动选择训练硬件

顺序：
1. `mps`（苹果芯片 GPU）
2. `cuda`（NVIDIA GPU）
3. `cpu`

### 4.2 `MLP` 类
角色：
- 定义神经网络结构（计算图）

### 4.3 `train_one_epoch(...)`
角色：
- 训练一个 epoch（完整跑完训练集一次）

每个 batch 的步骤：
1. 前向传播：`logits = model(data)`
2. 计算损失：`loss = criterion(logits, target)`
3. 反向传播：`loss.backward()`
4. 更新参数：`optimizer.step()`

### 4.4 `evaluate(...)`
角色：
- 在测试集上评估准确率
- 不更新参数（`@torch.no_grad()`）

### 4.5 `main()`
角色：
- 负责把“数据、模型、训练、评估、保存”串起来

---

## 5. 参数清单（每个参数的作用）

脚本可调参数：

- `--batch-size`（默认 128）
  - 一次喂给模型多少样本
  - 大：更快、更稳；太大可能显存不够
  - 小：更省内存；噪声更大

- `--epochs`（默认 12）
  - 全量训练轮数
  - 太小：欠拟合；太大：过拟合风险上升

- `--lr`（默认 8e-4）
  - 学习率，控制每步更新幅度
  - 太大：震荡甚至发散；太小：收敛慢

- `--hidden-dim`（默认 256）
  - 隐藏层宽度（模型容量）
  - 大：表达能力强；过大可能过拟合

- `--dropout`（默认 0.1）
  - 正则化强度
  - 大：抗过拟合更强；过大可能欠拟合

- `--weight-decay`（默认 1e-4）
  - L2 正则
  - 让权重不要过大，提升泛化

- `--data-dir`
  - MNIST 下载/读取目录

- `--save-path`
  - 最佳模型保存路径（按测试准确率）

---

## 6. 为什么这次调参这样设置

你现在这个“三层总结构”比之前模型更小。  
所以参数策略是：

- `hidden_dim=256`：给单隐藏层足够容量
- `batch_size=128`：训练速度和稳定性平衡
- `lr=8e-4` + `AdamW`：常见且稳定
- `dropout=0.1` + `weight_decay=1e-4`：轻度正则，防过拟合
- `epochs=12`：通常能到一个不错准确率

---

## 7. 怎么调参数（实战路线）

建议顺序：

1. 先固定：
   - `batch_size=128, lr=8e-4, hidden_dim=256, dropout=0.1`
2. 看训练/测试准确率差距：
   - 训练高、测试低：过拟合 -> 增大 `dropout` 或 `weight_decay`
   - 两者都低：欠拟合 -> 增大 `hidden_dim` 或 `epochs`
3. 再微调学习率：
   - 结果抖动大 -> 降低 `lr`
   - 收敛很慢 -> 略升 `lr`

常用尝试组合：
- A: `hidden_dim=128, dropout=0.1, lr=1e-3`
- B: `hidden_dim=256, dropout=0.1, lr=8e-4`（当前默认）
- C: `hidden_dim=512, dropout=0.2, lr=6e-4`

---

## 8. 最后跑出来的模型是什么

产物文件：
- `best_mlp_mnist.pth`

里面是什么：
- 模型参数（`state_dict`）
- 不是完整 Python 程序
- 需要同样的模型结构类 `MLP` 才能加载

保存逻辑：
- 每个 epoch 测试一次
- 只要测试准确率更高，就覆盖保存“最佳模型”

---

## 9. 怎么验证模型是否真的可用

### 9.1 训练过程验证

关注三项：
- `train loss` 是否下降
- `train acc` 是否上升
- `test acc` 是否稳定上升

### 9.2 模型文件验证

检查模型文件是否存在：

```bash
ls -lh /Users/ray/Documents/Fork/DeepLearning/best_mlp_mnist.pth
```

### 9.3 推理验证（抽样）

最基础思路：
- 从测试集拿一批图
- 用模型预测
- 看预测标签与真实标签是否一致

如果你需要，我可以再给你补一个 `predict_demo.py`，专门做单张图或批量可视化预测。

---

## 10. 运行命令

### 10.1 跑脚本

```bash
conda run -n ml_env python /Users/ray/Documents/Fork/DeepLearning/train_mnist_mlp.py
```

### 10.2 自定义参数

```bash
conda run -n ml_env python /Users/ray/Documents/Fork/DeepLearning/train_mnist_mlp.py \
  --epochs 15 \
  --batch-size 128 \
  --hidden-dim 256 \
  --dropout 0.1 \
  --lr 0.0008 \
  --weight-decay 0.0001
```

### 10.3 跑 Notebook

打开：
- `mnist_mlp_notebook.ipynb`

按顺序执行单元格即可。

---

## 11. 一句话总结

这版实现满足“总三层”要求，默认参数偏稳，适合课程作业与入门理解；后续调参主要围绕 `hidden_dim / lr / dropout / weight_decay / epochs` 这五个旋钮做平衡。
