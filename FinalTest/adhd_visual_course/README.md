# ADHD Visual Course

这是一个零依赖静态教材，直接打开 `index.html` 就能用。

## 文件结构

```text
adhd_visual_course/
├── index.html   # 页面骨架
├── styles.css   # 样式
├── app.js       # 课程内容、动画、交互逻辑
└── README.md    # 本说明
```

## 怎么打开

在文件管理器或浏览器里打开：

```text
FinalTest/adhd_visual_course/index.html
```

不需要安装 React、Next.js、Node 包或 Python 包。

## 现在覆盖的课程

- 这次 NER/RE 作业到底在训练什么
- Python 脚本、变量、函数、命令行参数
- JSON 数据、训练集、验证集、标签
- token、id、向量、tensor
- 模型家族：分类器、CNN、RNN、Transformer、LLM
- Attention 和 Transformer
- NER、BIOES、CRF、Precision/Recall/F1
- RE、实体对、NONE、neg_ratio、threshold
- batch_size、epochs、learning rate、patience 等调参

## 当前交互模块

- 左侧课程导航和进度条
- 每节课的依赖树
- 自下而上的分步解释
- 术语白话卡
- 代码显微镜：把概念映射回本项目脚本
- 常见误区提醒
- 对比表
- 项目映射
- 逻辑闭环小问题
- 主画布动画：pipeline、数据、tensor、attention、NER、RE、调参等
- 调参模拟器：显存压力、训练时间、过拟合风险
- 训练曲线动画：loss 下降、score 上升和过拟合风险

## 后续扩展建议

如果 `app.js` 继续变大，可以拆成：

```text
data/lessons.js        # 课程文字
visuals/canvas.js      # 动画绘制
visuals/tuning_lab.js  # 调参模拟器
```

当前版本为了方便直接上传和打开，先保持三文件结构。
