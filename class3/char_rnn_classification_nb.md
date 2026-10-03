{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 字符级 RNN 姓名分类教程\n",
    "\n",
    "本教程基于 PyTorch 官方教程: **NLP From Scratch: Classifying Names with a Character-Level RNN**\n",
    "\n",
    "我们将构建并训练一个基础的字符级循环神经网络（RNN），根据姓名的拼写预测其所属的语言/国家。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 导入必要的库"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PyTorch 版本: 2.5.1\n",
      "CUDA 可用: False\n",
      "MPS 可用: True\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import random\n",
    "import string\n",
    "import unicodedata\n",
    "import os\n",
    "import glob\n",
    "import time\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as ticker\n",
    "\n",
    "# 设置随机种子，保证结果可复现\n",
    "torch.manual_seed(42)\n",
    "random.seed(42)\n",
    "\n",
    "print(f\"PyTorch 版本: {torch.__version__}\")\n",
    "print(f\"CUDA 可用: {torch.cuda.is_available()}\")\n",
    "print(f\"MPS 可用: {torch.backends.mps.is_available()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 设备配置\n",
    "\n",
    "自动检测是否有加速设备，优先级: CUDA > MPS > CPU。\n",
    "- **CUDA**: NVIDIA GPU\n",
    "- **MPS**: Apple Silicon GPU (Mac M1/M2/M3)\n",
    "- **CPU**: 默认 fallback"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "使用设备: mps\n"
     ]
    }
   ],
   "source": [
    "# 自动检测加速设备: CUDA > MPS > CPU\n",
    "if torch.cuda.is_available():\n",
    "    device = torch.device('cuda')\n",
    "elif torch.backends.mps.is_available():\n",
    "    device = torch.device('mps')\n",
    "else:\n",
    "    device = torch.device('cpu')\n",
    "\n",
    "print(f\"使用设备: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 数据预处理\n",
    "\n",
    "### 3.1 字符集定义\n",
    "\n",
    "我们使用以下字符集来表示所有可能的输入字符：\n",
    "- 大小写英文字母 (a-z, A-Z)\n",
    "- 空格和标点符号 ( .,;')\n",
    "- 下划线 _ (用于表示未知字符)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "字符集大小: 58\n",
      "字符集: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'_\n"
     ]
    }
   ],
   "source": [
    "# 允许的字符集\n",
    "ALLOWED_CHARS = string.ascii_letters + \" .,;'\" + \"_\"\n",
    "N_LETTERS = len(ALLOWED_CHARS)\n",
    "N_HIDDEN = 128  # RNN 隐藏层大小\n",
    "\n",
    "print(f\"字符集大小: {N_LETTERS}\")\n",
    "print(f\"字符集: {ALLOWED_CHARS}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 Unicode 转 ASCII\n",
    "\n",
    "姓名数据包含各种 Unicode 字符（如重音符号），我们需要将其转换为纯 ASCII，以简化模型的输入层。\n",
    "\n",
    "例如: **'Ślusàrski'** → **'Slusarski'**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "'Ślusàrski' -> 'Slusarski'\n",
      "'François' -> 'Francois'\n",
      "' Müller ' -> 'Muller'\n"
     ]
    }
   ],
   "source": [
    "def unicode_to_ascii(s):\n",
    "    \"\"\"\n",
    "    将 Unicode 字符串转换为 ASCII\n",
    "    去掉重音符号，只保留允许的字符\n",
    "    \"\"\"\n",
    "    return ''.join(\n",
    "        c for c in unicodedata.normalize('NFD', s)\n",
    "        if unicodedata.category(c) != 'Mn'  # 去掉重音符号\n",
    "        and c in ALLOWED_CHARS\n",
    "    )\n",
    "\n",
    "# 测试\n",
    "print(f\"'Ślusàrski' -> '{unicode_to_ascii('Ślusàrski')}'\")\n",
    "print(f\"'François' -> '{unicode_to_ascii('François')}'\")\n",
    "print(f\"' Müller ' -> '{unicode_to_ascii('Müller')}'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.3 字符与 Tensor 的转换\n",
    "\n",
    "为了将字符输入神经网络，我们需要：\n",
    "1. 将每个字符转换为索引（0 到 N_LETTERS-1）\n",
    "2. 将索引转换为 one-hot 向量（只有一个位置为 1，其余为 0）\n",
    "3. 将整条姓名转换为 Tensor 序列"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "字母 'a' 的 one-hot 编码:\n",
      "tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
      "          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
      "          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n",
      "          0., 0., 0., 0., 0., 0., 0.]]])\n",
      "\n",
      "姓名 'Ahn' 的 Tensor 形状: torch.Size([3, 1, 58])\n"
     ]
    }
   ],
   "source": [
    "def letter_to_index(letter):\n",
    "    \"\"\"将字符转换为索引\"\"\"\n",
    "    if letter not in ALLOWED_CHARS:\n",
    "        return ALLOWED_CHARS.find(\"_\")  # 未知字符用下划线代替\n",
    "    return ALLOWED_CHARS.find(letter)\n",
    "\n",
    "def line_to_tensor(line):\n",
    "    \"\"\"\n",
    "    将姓名字符串转换为 Tensor\n",
    "    形状: (姓名长度, 1, 字符集大小)\n",
    "    每个字符是一个 one-hot 向量\n",
    "    \"\"\"\n",
    "    tensor = torch.zeros(len(line), 1, N_LETTERS, device=device)\n",
    "    for li, letter in enumerate(line):\n",
    "        idx = letter_to_index(letter)\n",
    "        tensor[li][0][idx] = 1\n",
    "    return tensor\n",
    "\n",
    "# 测试\n",
    "print(\"字母 'a' 的 one-hot 编码:\")\n",
    "print(line_to_tensor('a'))\n",
    "print(f\"\\n姓名 'Ahn' 的 Tensor 形状: {line_to_tensor('Ahn').shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 加载数据\n",
    "\n",
    "数据位于 `data/names/` 目录下，包含 18 个文本文件，每个文件以一种语言命名，包含该语言的姓氏列表。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "加载了 18 种语言的数据\n",
      "\n",
      "各语言样本数量:\n",
      "  Arabic         : 2000 个姓名\n",
      "  Chinese        :  268 个姓名\n",
      "  Czech          :  519 个姓名\n",
      "  Dutch          :  297 个姓名\n",
      "  English        : 3668 个姓名\n",
      "  French         :  277 个姓名\n",
      "  German         :  724 个姓名\n",
      "  Greek          :  203 个姓名\n",
      "  Irish          :  232 个姓名\n",
      "  Italian        :  709 个姓名\n",
      "  Japanese       :  991 个姓名\n",
      "  Korean         :   94 个姓名\n",
      "  Polish         :  139 个姓名\n",
      "  Portuguese     :   74 个姓名\n",
      "  Russian        : 9408 个姓名\n",
      "  Scottish       :  100 个姓名\n",
      "  Spanish        :  298 个姓名\n",
      "  Vietnamese     :   73 个姓名\n"
     ]
    }
   ],
   "source": [
    "def find_files(path):\n",
    "    \"\"\"查找所有 .txt 文件\"\"\"\n",
    "    return glob.glob(path)\n",
    "\n",
    "def read_lines(filename):\n",
    "    \"\"\"读取文件中的所有姓名\"\"\"\n",
    "    with open(filename, encoding='utf-8') as f:\n",
    "        lines = f.read().strip().split('\\n')\n",
    "    return [unicode_to_ascii(line) for line in lines]\n",
    "\n",
    "# 加载数据\n",
    "data_path = os.path.join('data', 'names', '*.txt')\n",
    "all_files = find_files(data_path)\n",
    "\n",
    "all_categories = []\n",
    "category_lines = {}\n",
    "\n",
    "for filename in all_files:\n",
    "    category = os.path.splitext(os.path.basename(filename))[0]\n",
    "    all_categories.append(category)\n",
    "    lines = read_lines(filename)\n",
    "    category_lines[category] = lines\n",
    "\n",
    "N_CATEGORIES = len(all_categories)\n",
    "\n",
    "print(f\"加载了 {N_CATEGORIES} 种语言的数据\\n\")\n",
    "print(\"各语言样本数量:\")\n",
    "for category in sorted(all_categories):\n",
    "    print(f\"  {category:15s}: {len(category_lines[category]):4d} 个姓名\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 探索数据\n",
    "\n",
    "让我们看看每个语言的一些示例姓名。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Arabic:\n",
      "  Khoury, Nahas, Daher, Gerges, Nazari\n",
      "\n",
      "Chinese:\n",
      "  Ang, AuYong, Bai, Ban, Bao\n",
      "\n",
      "Czech:\n",
      "  Abl, Adsit, Ajdrna, Alt, Antonowitsch\n",
      "\n",
      "Dutch:\n",
      "  Aalsburg, Aalst, Aarle, Achteren, Achthoven\n",
      "\n",
      "English:\n",
      "  Abbas, Abbey, Abbott, Abdi, Abel\n",
      "\n",
      "French:\n",
      "  Abel, Abraham, Adam, Albert, Allard\n"
     ]
    }
   ],
   "source": [
    "# 显示每个语言的前 5 个姓名\n",
    "for category in sorted(all_categories)[:6]:  # 只显示前 6 种\n",
    "    print(f\"\\n{category}:\")\n",
    "    print(\"  \" + \", \".join(category_lines[category][:5]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 定义 RNN 模型\n",
    "\n",
    "### 6.1 RNN 结构说明\n",
    "\n",
    "我们的 RNN 模型结构如下：\n",
    "\n",
    "```\n",
    "输入字符 (one-hot)     隐藏状态 (hidden)\n",
    "       │                    │\n",
    "       └────────┬───────────┘\n",
    "                │\n",
    "           [拼接 concat]\n",
    "                │\n",
    "        ┌───────┴───────┐\n",
    "        │               │\n",
    "   i2h (线性层)     i2o (线性层)\n",
    "        │               │\n",
    "    tanh 激活       LogSoftmax\n",
    "        │               │\n",
    "   新隐藏状态        输出 (类别概率)\n",
    "```\n",
    "\n",
    "**关键概念**：\n",
    "- **隐藏状态 (Hidden State)**：RNN 的\"记忆\"，包含之前所有字符的信息\n",
    "- **One-hot 编码**：每个字符用一个向量表示，只有对应位置为 1\n",
    "- **LogSoftmax**：将输出转换为对数概率分布"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RNN(\n",
      "  (i2h): Linear(in_features=186, out_features=128, bias=True)\n",
      "  (i2o): Linear(in_features=186, out_features=18, bias=True)\n",
      "  (softmax): LogSoftmax(dim=1)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "class RNN(nn.Module):\n",
    "    \"\"\"\n",
    "    字符级 RNN 分类器\n",
    "    \n",
    "    参数:\n",
    "        input_size: 输入大小 (字符集大小)\n",
    "        hidden_size: 隐藏层大小\n",
    "        output_size: 输出大小 (类别数量)\n",
    "    \"\"\"\n",
    "    \n",
    "    def __init__(self, input_size, hidden_size, output_size):\n",
    "        super(RNN, self).__init__()\n",
    "        self.hidden_size = hidden_size\n",
    "        \n",
    "        # 输入 + 隐藏状态 -> 新的隐藏状态\n",
    "        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)\n",
    "        # 输入 + 隐藏状态 -> 输出\n",
    "        self.i2o = nn.Linear(input_size + hidden_size, output_size)\n",
    "        # LogSoftmax 用于输出概率分布\n",
    "        self.softmax = nn.LogSoftmax(dim=1)\n",
    "    \n",
    "    def forward(self, input_tensor, hidden):\n",
    "        \"\"\"前向传播\"\"\"\n",
    "        # 拼接输入和隐藏状态\n",
    "        combined = torch.cat((input_tensor, hidden), 1)\n",
    "        # 计算新的隐藏状态\n",
    "        hidden = self.i2h(combined)\n",
    "        hidden = torch.tanh(hidden)\n",
    "        # 计算输出\n",
    "        output = self.i2o(combined)\n",
    "        output = self.softmax(output)\n",
    "        return output, hidden\n",
    "    \n",
    "    def init_hidden(self):\n",
    "        \"\"\"初始化隐藏状态为全零\"\"\"\n",
    "        return torch.zeros(1, self.hidden_size, device=device)\n",
    "\n",
    "# 创建模型实例\n",
    "rnn = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES).to(device)\n",
    "print(rnn)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 理解 RNN 的前向传播\n",
    "\n",
    "让我们手动运行一步，看看 RNN 是如何工作的。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "输入形状: torch.Size([1, 58])\n",
      "隐藏状态形状: torch.Size([1, 128])\n",
      "输出形状: torch.Size([1, 18]) (18 个类别的对数概率)\n",
      "输出示例 (前 5 个): tensor([-2.8870, -2.9534, -2.8236, -2.9104, -2.8127], grad_fn=<SliceBackward0>)\n"
     ]
    }
   ],
   "source": [
    "# 测试: 输入一个字符\n",
    "input_letter = line_to_tensor('A')[0]  # 字母 'A' 的 one-hot 向量\n",
    "hidden = rnn.init_hidden()\n",
    "\n",
    "output, next_hidden = rnn(input_letter, hidden)\n",
    "\n",
    "print(f\"输入形状: {input_letter.shape}\")\n",
    "print(f\"隐藏状态形状: {hidden.shape}\")\n",
    "print(f\"输出形状: {output.shape} (18 个类别的对数概率)\")\n",
    "print(f\"输出示例 (前 5 个): {output[0][:5]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. 准备训练\n",
    "\n",
    "### 7.1 辅助函数"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "类别: Japanese\n",
      "姓名: Araki\n",
      "类别 Tensor: tensor([3])\n",
      "姓名 Tensor 形状: torch.Size([5, 1, 58])\n"
     ]
    }
   ],
   "source": [
    "def random_training_example():\n",
    "    \"\"\"\n",
    "    随机选择一个训练样本\n",
    "    返回: (类别名, 姓名, 类别_tensor, 姓名_tensor)\n",
    "    \"\"\"\n",
    "    category = random.choice(all_categories)\n",
    "    line = random.choice(category_lines[category])\n",
    "    category_tensor = torch.tensor(\n",
    "        [all_categories.index(category)], dtype=torch.long, device=device\n",
    "    )\n",
    "    line_tensor = line_to_tensor(line)\n",
    "    return category, line, category_tensor, line_tensor\n",
    "\n",
    "def category_from_output(output):\n",
    "    \"\"\"从模型输出中获取预测的类别\"\"\"\n",
    "    top_n, top_i = output.topk(1)\n",
    "    category_i = top_i[0].item()\n",
    "    return all_categories[category_i], category_i\n",
    "\n",
    "def time_since(since):\n",
    "    \"\"\"计算已经过去的时间\"\"\"\n",
    "    now = time.time()\n",
    "    s = now - since\n",
    "    m = math.floor(s / 60)\n",
    "    s -= m * 60\n",
    "    return f'{m}m {s:.0f}s'\n",
    "\n",
    "# 测试随机采样\n",
    "category, line, category_tensor, line_tensor = random_training_example()\n",
    "print(f\"类别: {category}\")\n",
    "print(f\"姓名: {line}\")\n",
    "print(f\"类别 Tensor: {category_tensor}\")\n",
    "print(f\"姓名 Tensor 形状: {line_tensor.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.2 损失函数和优化器"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "损失函数: NLLLoss()\n",
      "学习率: 0.005\n"
     ]
    }
   ],
   "source": [
    "# 损失函数: 负对数似然损失 (Negative Log Likelihood Loss)\n",
    "# 适合配合 LogSoftmax 使用\n",
    "criterion = nn.NLLLoss()\n",
    "\n",
    "# 学习率\n",
    "learning_rate = 0.005\n",
    "\n",
    "print(f\"损失函数: {criterion}\")\n",
    "print(f\"学习率: {learning_rate}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7.3 单次训练步骤"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "姓名: Dubhan\n",
      "真实类别: Irish\n",
      "预测类别: Scottish\n",
      "损失: 2.9480\n"
     ]
    }
   ],
   "source": [
    "def train(category_tensor, line_tensor):\n",
    "    \"\"\"\n",
    "    训练一个样本\n",
    "    \n",
    "    步骤:\n",
    "    1. 初始化隐藏状态\n",
    "    2. 逐个字符输入 RNN\n",
    "    3. 计算损失\n",
    "    4. 反向传播\n",
    "    5. 更新参数\n",
    "    \"\"\"\n",
    "    # 初始化隐藏状态\n",
    "    hidden = rnn.init_hidden()\n",
    "    \n",
    "    # 梯度清零\n",
    "    rnn.zero_grad()\n",
    "    \n",
    "    # 逐个字符输入 RNN\n",
    "    for i in range(line_tensor.size()[0]):\n",
    "        output, hidden = rnn(line_tensor[i], hidden)\n",
    "    \n",
    "    # 计算损失 (使用最后一个输出)\n",
    "    loss = criterion(output, category_tensor)\n",
    "    \n",
    "    # 反向传播\n",
    "    loss.backward()\n",
    "    \n",
    "    # 手动更新参数 (梯度下降)\n",
    "    for p in rnn.parameters():\n",
    "        p.data.add_(p.grad.data, alpha=-learning_rate)\n",
    "    \n",
    "    return output, loss.item()\n",
    "\n",
    "# 测试训练一步\n",
    "category, line, category_tensor, line_tensor = random_training_example()\n",
    "output, loss = train(category_tensor, line_tensor)\n",
    "guess, guess_i = category_from_output(output)\n",
    "\n",
    "print(f\"姓名: {line}\")\n",
    "print(f\"真实类别: {category}\")\n",
    "print(f\"预测类别: {guess}\")\n",
    "print(f\"损失: {loss:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. 训练模型\n",
    "\n",
    "现在我们来训练模型。我们会迭代 100,000 次，每次随机选择一个样本进行训练。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "开始训练 100000 轮...\n",
      "\n",
      "0m 3s (  5000  5%) 损失: 2.8074 | Rompaye         -> Italian    ✗ (Dutch)\n",
      "0m 6s ( 10000 10%) 损失: 2.6700 | Medeiros        -> Greek      ✗ (Portuguese)\n",
      "0m 10s ( 15000 15%) 损失: 2.7543 | Coomber         -> French     ✗ (English)\n",
      "0m 13s ( 20000 20%) 损失: 0.7387 | Ghannam         -> Arabic     ✓\n",
      "0m 16s ( 25000 25%) 损失: 1.8387 | Hoang           -> Vietnamese ✓\n",
      "0m 20s ( 30000 30%) 损失: 4.6374 | Young           -> Chinese    ✗ (Scottish)\n",
      "0m 23s ( 35000 35%) 损失: 2.1867 | Lokay           -> Scottish   ✗ (Czech)\n"
     ]
    }
   ],
   "source": [
    "# 训练参数\n",
    "n_iters = 100000\n",
    "print_every = 5000\n",
    "plot_every = 1000\n",
    "\n",
    "# 记录数据\n",
    "current_loss = 0\n",
    "all_losses = []\n",
    "all_accuracies = []\n",
    "start = time.time()\n",
    "\n",
    "print(f\"开始训练 {n_iters} 轮...\\n\")\n",
    "\n",
    "for iter in range(1, n_iters + 1):\n",
    "    category, line, category_tensor, line_tensor = random_training_example()\n",
    "    output, loss = train(category_tensor, line_tensor)\n",
    "    current_loss += loss\n",
    "    \n",
    "    # 打印进度\n",
    "    if iter % print_every == 0:\n",
    "        guess, guess_i = category_from_output(output)\n",
    "        correct = '✓' if guess == category else f'✗ ({category})'\n",
    "        print(f'{time_since(start)} ({iter:6d} {iter/n_iters*100:2.0f}%) '\n",
    "              f'损失: {loss:.4f} | {line:15s} -> {guess:10s} {correct}')\n",
    "    \n",
    "    # 记录损失\n",
    "    if iter % plot_every == 0:\n",
    "        all_losses.append(current_loss / plot_every)\n",
    "        current_loss = 0\n",
    "\n",
    "print(f\"\\n训练完成! 总用时: {time_since(start)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. 可视化训练过程"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 绘制损失曲线\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(all_losses)\n",
    "plt.title('训练损失曲线')\n",
    "plt.xlabel('每 1000 次迭代')\n",
    "plt.ylabel('平均损失')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. 评估模型\n",
    "\n",
    "### 10.1 计算准确率\n",
    "\n",
    "我们使用混淆矩阵来评估模型在各个语言上的表现。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate(line_tensor):\n",
    "    \"\"\"评估一个姓名，返回模型输出\"\"\"\n",
    "    hidden = rnn.init_hidden()\n",
    "    with torch.no_grad():\n",
    "        for i in range(line_tensor.size()[0]):\n",
    "            output, hidden = rnn(line_tensor[i], hidden)\n",
    "    return output\n",
    "\n",
    "# 构建混淆矩阵\n",
    "confusion = torch.zeros(N_CATEGORIES, N_CATEGORIES)\n",
    "n_confusion = 10000\n",
    "\n",
    "for i in range(n_confusion):\n",
    "    category, line, category_tensor, line_tensor = random_training_example()\n",
    "    output = evaluate(line_tensor)\n",
    "    guess, guess_i = category_from_output(output)\n",
    "    category_i = all_categories.index(category)\n",
    "    confusion[category_i][guess_i] += 1\n",
    "\n",
    "# 归一化 (按行)\n",
    "for i in range(N_CATEGORIES):\n",
    "    confusion[i] = confusion[i] / confusion[i].sum()\n",
    "\n",
    "# 计算总体准确率\n",
    "accuracy = sum(confusion[i][i] for i in range(N_CATEGORIES)) / N_CATEGORIES * 100\n",
    "print(f\"总体准确率: {accuracy:.1f}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 10.2 绘制混淆矩阵"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 绘制混淆矩阵\n",
    "fig = plt.figure(figsize=(12, 10))\n",
    "ax = fig.add_subplot(111)\n",
    "cax = ax.matshow(confusion.cpu().numpy(), cmap='Blues')\n",
    "fig.colorbar(cax)\n",
    "\n",
    "# 设置刻度\n",
    "ax.set_xticks(range(N_CATEGORIES))\n",
    "ax.set_yticks(range(N_CATEGORIES))\n",
    "\n",
    "# 设置标签 (旋转 90 度避免重叠)\n",
    "ax.set_xticklabels(sorted(all_categories), rotation=90, fontsize=8)\n",
    "ax.set_yticklabels(sorted(all_categories), fontsize=8)\n",
    "\n",
    "ax.set_xlabel('预测类别', fontsize=12)\n",
    "ax.set_ylabel('真实类别', fontsize=12)\n",
    "ax.set_title('混淆矩阵', fontsize=14, pad=20)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. 预测功能\n",
    "\n",
    "### 11.1 预测单个姓名"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(input_line, n_predictions=3):\n",
    "    \"\"\"\n",
    "    预测一个姓名所属的语言\n",
    "    \n",
    "    参数:\n",
    "        input_line: 输入的姓名\n",
    "        n_predictions: 返回前 N 个最可能的预测结果\n",
    "    \"\"\"\n",
    "    print(f\"\\n>>> {input_line}\")\n",
    "    with torch.no_grad():\n",
    "        output = evaluate(line_to_tensor(input_line))\n",
    "        \n",
    "        # 获取前 N 个预测\n",
    "        topv, topi = output.topk(n_predictions, 1, True)\n",
    "        predictions = []\n",
    "        \n",
    "        for i in range(n_predictions):\n",
    "            value = topv[0][i].item()\n",
    "            category_index = topi[0][i].item()\n",
    "            print(f'  ({math.exp(value)*100:5.1f}%) {all_categories[category_index]}')\n",
    "            predictions.append([value, all_categories[category_index]])\n",
    "        \n",
    "        return predictions\n",
    "\n",
    "# 测试一些姓名\n",
    "test_names = [\n",
    "    \"Zhang\",        # 中文\n",
    "    \"Smith\",        # 英文\n",
    "    \"Sato\",         # 日文\n",
    "    \"Kim\",          # 韩文\n",
    "    \"Martin\",       # 法文\n",
    "    \"Rossi\",        # 意大利文\n",
    "    \"Mueller\",      # 德文\n",
    "    \"Ivanov\",       # 俄文\n",
    "    \"Garcia\",       # 西班牙文\n",
    "    \"Nguyen\",       # 越南文\n",
    "]\n",
    "\n",
    "for name in test_names:\n",
    "    predict(name)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 11.2 交互式预测\n",
    "\n",
    "你可以输入任何姓名来测试模型！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 测试你自己的姓名\n",
    "predict(\"YourName\")\n",
    "predict(\"Einstein\")\n",
    "predict(\"Bach\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. 分析模型学到的模式\n",
    "\n",
    "让我们看看模型是否学到了一些有趣的模式。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 测试不同后缀的姓名\n",
    "print(\"===== 测试后缀模式 =====\")\n",
    "suffixes = {\n",
    "    \"-ski\": [\"Kowalski\", \"Nowakowski\", \"Jablonski\"],  # 波兰\n",
    "    \"-son\": [\"Johnson\", \"Jackson\", \"Wilson\"],        # 英文/北欧\n",
    "    \"-ez\": [\"Gonzalez\", \"Rodriguez\", \"Perez\"],       # 西班牙\n",
    "    \"-berg\": [\"Goldberg\", \"Silverberg\", \"Rosenberg\"], # 犹太/德\n",
    "    \"-ini\": [\"Rossini\", \"Puccini\", \"Bernini\"],       # 意大利\n",
    "}\n",
    "\n",
    "for suffix, names in suffixes.items():\n",
    "    print(f\"\\n后缀 '{suffix}':\")\n",
    "    for name in names:\n",
    "        predict(name, n_predictions=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. 保存和加载模型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 保存模型\n",
    "model_path = 'char_rnn_model.pth'\n",
    "torch.save({\n",
    "    'model_state_dict': rnn.state_dict(),\n",
    "    'all_categories': all_categories,\n",
    "    'n_hidden': N_HIDDEN,\n",
    "    'n_letters': N_LETTERS,\n",
    "}, model_path)\n",
    "print(f\"模型已保存到: {model_path}\")\n",
    "\n",
    "# 加载模型\n",
    "def load_model(path):\n",
    "    checkpoint = torch.load(path, map_location=device)\n",
    "    model = RNN(checkpoint['n_letters'], checkpoint['n_hidden'], len(checkpoint['all_categories']))\n",
    "    model.load_state_dict(checkpoint['model_state_dict'])\n",
    "    model.to(device)\n",
    "    return model, checkpoint['all_categories']\n",
    "\n",
    "# 测试加载\n",
    "loaded_rnn, loaded_categories = load_model(model_path)\n",
    "print(f\"模型加载成功!\")\n",
    "print(f\"类别数: {len(loaded_categories)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 14. 总结\n",
    "\n",
    "### 本教程学到的内容:\n",
    "\n",
    "1. **数据预处理**: 将文本数据转换为神经网络可以处理的 Tensor 格式\n",
    "2. **One-hot 编码**: 将离散字符表示为向量\n",
    "3. **RNN 结构**: 理解循环神经网络的基本原理和隐藏状态的概念\n",
    "4. **序列处理**: 如何逐个字符处理变长序列\n",
    "5. **分类任务**: 使用 NLLLoss 和 LogSoftmax 进行分类\n",
    "\n",
    "### 模型局限性:\n",
    "\n",
    "- 只使用字符信息，没有考虑发音\n",
    "- 对拼写相似的语言容易混淆 (如西班牙文和葡萄牙文)\n",
    "- 无法处理完全未知的字符模式\n",
    "\n",
    "### 改进方向:\n",
    "\n",
    "- 使用 LSTM 或 GRU 替代简单 RNN\n",
    "- 增加网络深度和隐藏层大小\n",
    "- 使用更复杂的数据增强技术\n",
    "- 尝试使用 Transformer 架构"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
