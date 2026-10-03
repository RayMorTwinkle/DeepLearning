# -*- coding: utf-8 -*-
"""
字符级 RNN 姓名分类
基于 PyTorch 官方教程: NLP From Scratch: Classifying Names with a Character-Level RNN

功能: 训练一个字符级循环神经网络，根据姓名的拼写预测其所属的语言/国家
"""

import torch
import torch.nn as nn
import random
import string
import unicodedata
import os
import glob
import time
import math

# ==================== 1. 设备配置 ====================
# 自动检测是否有 GPU，优先使用 CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 2. 数据预处理 ====================

# 允许的字符集: 大小写字母 + 空格 + 标点 + 下划线(用于未知字符)
ALLOWED_CHARS = string.ascii_letters + " .,;'" + "_"
N_LETTERS = len(ALLOWED_CHARS)
N_HIDDEN = 128  # RNN 隐藏层大小


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


def letter_to_index(letter):
    """
    将字符转换为索引，如果字符不在允许集合中则返回下划线索引
    """
    if letter not in ALLOWED_CHARS:
        return ALLOWED_CHARS.find("_")
    return ALLOWED_CHARS.find(letter)


def line_to_tensor(line):
    """
    将姓名字符串转换为 Tensor
    形状: (姓名长度, 1, 字符集大小)
    每个字符是一个 one-hot 向量
    """
    tensor = torch.zeros(len(line), 1, N_LETTERS, device=device)
    for li, letter in enumerate(line):
        idx = letter_to_index(letter)
        tensor[li][0][idx] = 1
    return tensor


def find_files(path):
    """查找所有 .txt 文件"""
    return glob.glob(path)


def read_lines(filename):
    """
    读取文件中的所有行，并将 Unicode 转换为 ASCII
    返回姓名列表
    """
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


# ==================== 3. 加载数据 ====================

# 数据路径
data_path = os.path.join(os.path.dirname(__file__), 'data', 'names', '*.txt')
all_files = find_files(data_path)

# 所有类别（语言）
all_categories = []
# 每个类别的姓名列表
category_lines = {}

for filename in all_files:
    # 从文件名提取类别名，例如 "Chinese.txt" -> "Chinese"
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

N_CATEGORIES = len(all_categories)
print(f"加载了 {N_CATEGORIES} 种语言的数据")
print(f"语言列表: {all_categories}")

# 统计每个语言的姓名数量
for category in all_categories:
    print(f"  {category}: {len(category_lines[category])} 个姓名")


# ==================== 4. 定义 RNN 模型 ====================

class RNN(nn.Module):
    """
    字符级 RNN 分类器
    
    结构:
    - 输入: one-hot 字符向量 (大小: N_LETTERS)
    - 隐藏层: 线性变换 + tanh 激活
    - 输出: 每个类别的分数 (大小: N_CATEGORIES)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 从输入和隐藏状态到新的隐藏状态的线性变换
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 从输入和隐藏状态到输出的线性变换
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # softmax 用于输出概率分布
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_tensor, hidden):
        """
        前向传播
        input_tensor: 当前字符的 one-hot 向量, 形状 (1, N_LETTERS)
        hidden: 上一时刻的隐藏状态, 形状 (1, N_HIDDEN)
        """
        # 将输入和隐藏状态拼接
        combined = torch.cat((input_tensor, hidden), 1)
        # 计算新的隐藏状态
        hidden = self.i2h(combined)
        hidden = torch.tanh(hidden)
        # 计算输出
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        """初始化隐藏状态"""
        return torch.zeros(1, self.hidden_size, device=device)


# 创建模型实例
rnn = RNN(N_LETTERS, N_HIDDEN, N_CATEGORIES).to(device)
print(f"\n模型结构:\n{rnn}")


# ==================== 5. 训练准备 ====================

def random_training_example():
    """
    随机选择一个训练样本
    返回: (类别, 类别索引, 姓名, 姓名tensor)
    """
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long, device=device)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def category_from_output(output):
    """
    从模型输出中获取预测的类别
    output: 模型输出, 形状 (1, N_CATEGORIES)
    返回: (预测类别名, 类别索引)
    """
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def time_since(since):
    """计算已经过去的时间"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:.0f}s'


# 损失函数: 负对数似然损失 (适合分类任务)
criterion = nn.NLLLoss()
# 学习率
learning_rate = 0.005


def train(category_tensor, line_tensor):
    """
    训练一个样本
    返回: (损失值, 预测类别名)
    """
    # 初始化隐藏状态
    hidden = rnn.init_hidden()
    
    # 梯度清零
    rnn.zero_grad()
    
    # 逐个字符输入 RNN
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    # 计算损失
    loss = criterion(output, category_tensor)
    
    # 反向传播
    loss.backward()
    
    # 手动更新参数 (梯度下降)
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
    
    return output, loss.item()


# ==================== 6. 训练模型 ====================

def run_training(n_iters=100000, print_every=5000, plot_every=1000):
    """
    运行训练
    n_iters: 总迭代次数
    print_every: 每隔多少次打印一次
    plot_every: 每隔多少次记录一次损失
    """
    current_loss = 0
    all_losses = []
    start = time.time()
    
    print(f"\n开始训练 {n_iters} 轮...")
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        
        # 打印进度
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else f'✗ ({category})'
            print(f'{time_since(start)} ({iter} {iter/n_iters*100:.0f}%) '
                  f'损失: {loss:.4f} | 姓名: {line} -> 预测: {guess} {correct}')
        
        # 记录损失用于绘图
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
    return all_losses


# 运行训练 (默认 10 万次迭代)
print("\n" + "="*50)
print("训练模型")
print("="*50)
all_losses = run_training(n_iters=100000)


# ==================== 7. 评估模型 ====================

def evaluate(line_tensor):
    """
    评估一个姓名，返回模型输出
    """
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output


def predict(input_line, n_predictions=3):
    """
    预测一个姓名所属的语言
    input_line: 输入的姓名
    n_predictions: 返回前 N 个最可能的预测结果
    """
    print(f"\n>>> {input_line}")
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))
        
        # 获取前 N 个预测
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []
        
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(f'({value:.2f}) {all_categories[category_index]}')
            predictions.append([value, all_categories[category_index]])
        
        return predictions


def evaluate_accuracy():
    """
    在测试集上评估模型准确率
    """
    confusion = torch.zeros(N_CATEGORIES, N_CATEGORIES)
    n_confusion = 10000
    
    # 随机选择样本进行评估
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example()
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    # 归一化
    for i in range(N_CATEGORIES):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    # 计算准确率
    correct = 0
    total = 0
    for i in range(N_CATEGORIES):
        correct += confusion[i][i]
        total += confusion[i].sum()
    accuracy = correct / total * 100
    
    print(f"\n模型准确率: {accuracy:.2f}%")
    return confusion


# 评估模型
print("\n" + "="*50)
print("评估模型")
print("="*50)
confusion = evaluate_accuracy()


# ==================== 8. 测试预测 ====================

print("\n" + "="*50)
print("测试预测")
print("="*50)

# 测试一些姓名
test_names = [
    "Zhang",      # 中文
    "Smith",      # 英文
    "Sato",       # 日文
    "Kim",        # 韩文
    "Martin",     # 法文
    "Rossi",      # 意大利文
    "Mueller",    # 德文
    "Ivanov",     # 俄文
    "Garcia",     # 西班牙文
    "Santos",     # 葡萄牙文
    "Nguyen",     # 越南文
    "Kowalski",   # 波兰文
    "Van Dijk",   # 荷兰文
    "O'Brien",    # 爱尔兰文
    "Papadopoulos", # 希腊文
    "Novak",      # 捷克文
    "Hansen",     # 丹麦/挪威
    "Ali",        # 阿拉伯
    "Cohen",      # 犹太/希伯来
    "MacDonald",  # 苏格兰
]

for name in test_names:
    predict(name)


# ==================== 9. 保存模型 ====================

def save_model(path='char_rnn_model.pth'):
    """保存模型"""
    torch.save({
        'model_state_dict': rnn.state_dict(),
        'all_categories': all_categories,
        'n_hidden': N_HIDDEN,
        'n_letters': N_LETTERS,
    }, path)
    print(f"\n模型已保存到: {path}")


def load_model(path='char_rnn_model.pth'):
    """加载模型"""
    checkpoint = torch.load(path, map_location=device)
    
    model = RNN(checkpoint['n_letters'], checkpoint['n_hidden'], len(checkpoint['all_categories']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, checkpoint['all_categories']


# 保存模型
save_model(os.path.join(os.path.dirname(__file__), 'char_rnn_model.pth'))


# ==================== 10. 可视化训练过程 (可选) ====================

def plot_losses(losses):
    """绘制训练损失曲线"""
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(losses)
        plt.title('训练损失')
        plt.xlabel('每 1000 次迭代')
        plt.ylabel('损失')
        plt.savefig('training_loss.png')
        print("\n损失曲线已保存到 training_loss.png")
        plt.show()
    except ImportError:
        print("\n未安装 matplotlib，跳过绘图")


if __name__ == '__main__':
    # 如果直接运行脚本，绘制损失曲线
    if len(all_losses) > 0:
        plot_losses(all_losses)
    
    # 交互式预测
    print("\n" + "="*50)
    print("交互式预测模式")
    print("输入一个姓名来预测其语言 (输入 'quit' 退出)")
    print("="*50)
    
    while True:
        user_input = input("\n请输入姓名: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input:
            predict(user_input)
