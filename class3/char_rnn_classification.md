# 字符级 RNN 姓名分类教程

> 基于 PyTorch 官方教程: **NLP From Scratch: Classifying Names with a Character-Level RNN**
>
> 作者: Sean Robertson | 最后更新: 2025-11-11

---

## 目录

1. [项目简介](#1-项目简介)
2. [环境准备](#2-环境准备)
3. [核心概念讲解](#3-核心概念讲解)
4. [代码详解](#4-代码详解)
5. [训练与评估](#5-训练与评估)
6. [结果分析](#6-结果分析)
7. [练习与扩展](#7-练习与扩展)

---

## 1. 项目简介

### 1.1 我们要做什么？

我们将构建一个**字符级循环神经网络（Character-Level RNN）**，它能够根据一个姓名的拼写，预测这个姓名属于哪种语言或国家。

**示例：**
| 姓名 | 预测结果 |
|------|---------|
| Zhang | Chinese |
| Smith | English |
| Sato | Japanese |
| Mueller | German |
| Rossi | Italian |

### 1.2 数据集

数据集位于 `data/names/` 目录下，包含 **18 种语言**的姓氏：

- **亚洲**: Chinese, Japanese, Korean, Vietnamese, Arabic
- **欧洲**: English, French, German, Italian, Spanish, Portuguese, Russian, Polish, Czech, Dutch, Greek, Irish, Scottish

每种语言一个 `.txt` 文件，每行一个姓名。

### 1.3 为什么用字符级 RNN？

传统方法处理文本需要复杂的特征工程（如 n-gram、TF-IDF）。RNN 的优势在于：

- **自动学习特征**: 不需要手动提取特征
- **处理变长序列**: 姓名长度不固定，RNN 可以处理
- **捕捉顺序信息**: 字符的排列顺序很重要（如 `-ski` 后缀常见于波兰语）

---

## 2. 环境准备

### 2.1 需要的库

```python
import torch
import torch.nn as nn
import random
import string
import unicodedata
import os
import glob
import time
import math
```

### 2.2 设备配置

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

自动检测 GPU，优先使用 CUDA 加速训练。

---

## 3. 核心概念讲解

### 3.1 什么是 RNN？

**RNN（Recurrent Neural Network，循环神经网络）** 是一种专门用于处理序列数据的神经网络。

**与普通神经网络的区别：**

```
普通神经网络: 输入 -> 隐藏层 -> 输出
RNN:          输入_t + 隐藏状态_{t-1} -> 隐藏状态_t -> 输出_t
                    ↑___________________________|
                           (循环连接)
```

**关键特性：**
- **隐藏状态（Hidden State）**: RNN 的"记忆"，包含之前所有输入的信息
- **共享参数**: 每个时间步使用相同的权重
- **变长处理**: 可以处理不同长度的序列

### 3.2 One-Hot 编码

神经网络只能处理数字，所以需要将字符转换为向量。

**One-Hot 编码规则：**
- 向量长度 = 字符集大小
- 只有对应字符的位置为 1，其余为 0

**示例（假设字符集为 "abc"）：**
```
'a' = [1, 0, 0]
'b' = [0, 1, 0]
'c' = [0, 0, 1]
```

### 3.3 我们的 RNN 结构

```
输入字符 (one-hot)     隐藏状态 (hidden)
       │                    │
       └────────┬───────────┘
                │
           [拼接 concat]
                │
        ┌───────┴───────┐
        │               │
   i2h (线性层)     i2o (线性层)
        │               │
    tanh 激活       LogSoftmax
        │               │
   新隐藏状态        输出 (类别概率)
```

**参数说明：**
- `i2h`: input to hidden，输入到隐藏状态的变换
- `i2o`: input to output，输入到输出的变换
- `hidden_size`: 隐藏状态维度（我们设为 128）

---

## 4. 代码详解

### 4.1 数据预处理

#### 4.1.1 Unicode 转 ASCII

姓名数据包含各种 Unicode 字符（如重音符号），需要转换为纯 ASCII。

```python
def unicode_to_ascii(s):
    """
    将 Unicode 字符串转换为 ASCII
    例如: 'Ślusàrski' -> 'Slusarski'
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'  # 去掉重音符号
        and c in ALLOWED_CHARS
    )
```

**原理：**
- `unicodedata.normalize('NFD', s)`: 将字符分解为基本字符和重音符号
- `unicodedata.category(c) != 'Mn'`: 过滤掉重音符号（Mark, nonspacing）

#### 4.1.2 字符转 Tensor

```python
def line_to_tensor(line):
    """
    将姓名字符串转换为 Tensor
    形状: (姓名长度, 1, 字符集大小)
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS, device=device)
    for li, letter in enumerate(line):
        idx = letter_to_index(letter)
        tensor[li][0][idx] = 1  # one-hot 编码
    return tensor
```

**为什么形状是 `(length, 1, N_LETTERS)`？**
- `length`: 姓名长度（序列长度）
- `1`: batch size（每次处理一个姓名）
- `N_LETTERS`: 字符集大小（one-hot 维度）

### 4.2 RNN 模型定义

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 输入 + 隐藏状态 -> 新的隐藏状态
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 输入 + 隐藏状态 -> 输出
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # LogSoftmax 用于输出概率分布
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_tensor, hidden):
        # 拼接输入和隐藏状态
        combined = torch.cat((input_tensor, hidden), 1)
        # 计算新的隐藏状态
        hidden = self.i2h(combined)
        hidden = torch.tanh(hidden)
        # 计算输出
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        # 初始化隐藏状态为全零
        return torch.zeros(1, self.hidden_size, device=device)
```

**关键理解：**

1. **为什么拼接输入和隐藏状态？**
   - RNN 需要同时考虑"当前看到什么"（输入）和"之前记住了什么"（隐藏状态）

2. **为什么用 tanh 激活？**
   - tanh 将值压缩到 (-1, 1)，帮助控制梯度大小，防止梯度爆炸/消失

3. **为什么用 LogSoftmax？**
   - 将输出转换为对数概率，配合 NLLLoss 使用

### 4.3 训练过程

#### 4.3.1 单次训练步骤

```python
def train(category_tensor, line_tensor):
    # 1. 初始化隐藏状态
    hidden = rnn.init_hidden()
    
    # 2. 梯度清零
    rnn.zero_grad()
    
    # 3. 逐个字符输入 RNN
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    # 4. 计算损失（只使用最后一个输出）
    loss = criterion(output, category_tensor)
    
    # 5. 反向传播
    loss.backward()
    
    # 6. 更新参数（手动梯度下降）
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    
    return output, loss.item()
```

**训练流程图：**

```
姓名 "Smith"
   │
   ▼
'S' -> RNN -> 隐藏状态_1
'm' -> RNN -> 隐藏状态_2
'i' -> RNN -> 隐藏状态_3
't' -> RNN -> 隐藏状态_4
'h' -> RNN -> 输出 (18 个类别的概率)
   │
   ▼
与真实标签比较 -> 计算损失 -> 反向传播 -> 更新权重
```

#### 4.3.2 为什么只使用最后一个输出？

因为我们做的是**序列到单个标签**的分类任务。RNN 在处理完所有字符后，隐藏状态包含了整个姓名的信息，所以最后一个输出就足以做出判断。

### 4.4 损失函数

```python
criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
```

**为什么用 NLLLoss？**

1. 我们的模型输出的是 **LogSoftmax**（对数概率）
2. NLLLoss 期望输入是对数概率
3. 数学上等价于 CrossEntropyLoss，但更高效

**直观理解：**
- 如果模型对正确类别的预测概率高 → 损失小
- 如果模型对正确类别的预测概率低 → 损失大

---

## 5. 训练与评估

### 5.1 运行训练

```bash
python char_rnn_classification.py
```

**训练输出示例：**
```
使用设备: cuda:0
加载了 18 种语言的数据

开始训练 100000 轮...

0m 3s (  5000   5%) 损失: 1.2345 | Zhang           -> Chinese   ✓
0m 6s ( 10000  10%) 损失: 0.8765 | Smith           -> English   ✓
...
```

### 5.2 训练技巧

1. **随机采样**: 每次随机选择一个样本，避免模型记忆顺序
2. **学习率**: 0.005 是一个较好的起点，太大容易震荡，太小收敛慢
3. **迭代次数**: 10 万次迭代通常能达到较好的效果

### 5.3 评估指标

**准确率（Accuracy）**: 正确预测的样本数 / 总样本数

**混淆矩阵**: 展示每个类别的预测情况，帮助发现模型在哪些语言上表现不好

---

## 6. 结果分析

### 6.1 预期准确率

训练 10 万轮后，模型准确率通常在 **75-85%** 之间。

### 6.2 容易混淆的语言

| 语言对 | 原因 |
|--------|------|
| Spanish ↔ Portuguese | 同属罗曼语族，拼写相似 |
| English ↔ Scottish | 地理位置接近，姓氏混合 |
| Czech ↔ Polish | 同属斯拉夫语族 |

### 6.3 模型学到的模式

**后缀模式：**
- `-ski`: 波兰语 (Kowalski)
- `-son`: 英语/北欧语 (Johnson)
- `-ez`: 西班牙语 (Gonzalez)
- `-ini`: 意大利语 (Rossini)
- `-berg`: 德语/犹太语 (Goldberg)

**前缀模式：**
- `Mc-`, `Mac-`: 苏格兰/爱尔兰语 (McDonald)
- `O'`: 爱尔兰语 (O'Brien)
- `Van `: 荷兰语 (Van Dijk)

---

## 7. 练习与扩展

### 7.1 基础练习

1. **修改隐藏层大小**: 尝试 64, 256, 512，观察对结果的影响
2. **调整学习率**: 尝试 0.001, 0.01，观察训练稳定性
3. **增加训练轮数**: 尝试 20 万次迭代，观察准确率是否提升

### 7.2 进阶扩展

1. **使用 LSTM**: 将简单 RNN 替换为 LSTM，解决长序列依赖问题
   ```python
   self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
   ```

2. **使用 GRU**: GRU 是 LSTM 的简化版，参数更少
   ```python
   self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
   ```

3. **双向 RNN**: 同时考虑正向和反向的字符序列
   ```python
   self.birnn = nn.RNN(input_size, hidden_size, bidirectional=True)
   ```

4. **注意力机制**: 让模型关注姓名中最重要的字符

### 7.3 可视化探索

1. **绘制损失曲线**: 观察训练是否收敛
2. **绘制混淆矩阵**: 发现模型的弱点
3. **隐藏状态可视化**: 使用 t-SNE 降维，观察不同语言的隐藏状态分布

---

## 附录：完整文件列表

```
class3/
├── data/
│   └── names/              # 18 种语言的姓名数据
│       ├── Chinese.txt
│       ├── English.txt
│       ├── Japanese.txt
│       └── ...
├── char_rnn_classification.py      # Python 脚本（可直接运行）
├── char_rnn_classification.ipynb   # Jupyter Notebook（交互式学习）
├── char_rnn_classification.md      # 本教程文档
└── char_rnn_model.pth              # 训练好的模型（运行后生成）
```

## 参考资源

- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

> **祝你学习愉快！** 如果有任何问题，欢迎随时提问。
