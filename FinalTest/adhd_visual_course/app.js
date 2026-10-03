const lessons = [
  {
    tag: "总地图",
    title: "你这次到底在训练什么？",
    oneLine: "把一段农业论文文本，变成“实体清单 + 关系清单”。",
    deps: [
      "文本是一串字符，不是模型能直接吃的东西",
      "标签是老师给模型看的标准答案",
      "训练是反复做题、对答案、改错",
      "NER 先找名词片段，RE 再判断片段之间有没有关系",
    ],
    steps: [
      ["第一层：文本", "一句话像一条长纸带，上面写着 sorghum、gene、drought tolerance。"],
      ["第二层：实体", "NER（命名实体识别：给文字片段贴名字）会圈出“作物、基因、性状”。"],
      ["第三层：关系", "RE（关系抽取：判断两个圈之间是什么箭头）会判断“基因影响性状”。"],
      ["第四层：提交物", "predictions.json 就是模型交卷：每条文本里圈了什么、连了什么箭头。"],
    ],
    table: [
      ["东西", "生活类比", "你项目里的名字", "产物"],
      ["文本", "原始考卷", "HW_train_data.json 的 text", "字符串"],
      ["NER", "拿荧光笔圈重点", "train_ner.py", "output_ner/best_ner_model.pt"],
      ["RE", "给重点之间画箭头", "train_re.py", "output_re/best_re_model.pt"],
      ["推理", "正式交卷", "predict.py", "predictions.json"],
    ],
    bridge: "你的项目不是“让模型聊天”，而是让模型做结构化阅读：先圈词，再连线。",
    quiz: "如果文本里有 sorghum 和 drought tolerance，哪个模块负责先把它们圈出来？",
    visual: "pipeline",
  },
  {
    tag: "Python",
    title: "Python 不是魔法，是菜谱执行器",
    oneLine: "脚本就是菜谱，变量是贴了名字的小碗，函数是一台固定动作的小机器。",
    deps: [
      "文件路径：告诉电脑原料在哪里",
      "变量：给一个东西起名字，之后能重复使用",
      "函数：把一套动作打包，输入不同材料，输出不同结果",
      "命令行参数：从外面递给脚本的旋钮",
    ],
    steps: [
      ["变量", "`batch_size = 12` 就像碗上贴纸：这碗叫 batch_size，里面装数字 12。"],
      ["函数", "`load_data(path)` 像榨汁机：放入文件路径，吐出 Python 能处理的数据。"],
      ["脚本", "`train_ner.py` 是一张完整菜谱：读数据、建模型、训练、保存。"],
      ["参数", "`--epochs 8` 是菜谱旁边的旋钮：这锅饭翻炒 8 轮。"],
    ],
    table: [
      ["Python 词", "白话", "本项目例子", "你该怎么读"],
      ["变量", "贴了名字的盒子", "`args.batch_size`", "现在 batch 是多少"],
      ["函数", "固定动作机器", "`prepare_re_samples`", "把数据变成 RE 样本"],
      ["类", "机器蓝图", "`NERModel`", "定义模型长什么样"],
      ["参数", "命令行旋钮", "`--lr 1e-5`", "控制训练脾气"],
    ],
    bridge: "你不用先精通 Python 才能读项目。先认出“输入、处理、输出”三段，就能看懂大半。",
    quiz: "`--batch_size 12` 更像代码里的“原料”，还是“旋钮”？",
    visual: "recipe",
  },
  {
    tag: "数据",
    title: "JSON 数据像一叠带答案的阅读理解",
    oneLine: "每条样本都有 text、entities、relations，分别是题目、圈词答案、连线答案。",
    deps: [
      "JSON 是一种规整的文本盒子",
      "训练集是带标准答案的练习册",
      "验证集是小测验，用来判断是不是学偏了",
      "标签不均衡会让模型偏爱常见答案",
    ],
    steps: [
      ["text", "原文，好比阅读理解的文章。"],
      ["entities", "老师已经圈好的词：从第几个字符开始，到第几个字符结束，属于什么类型。"],
      ["relations", "老师已经画好的箭头：哪个实体指向哪个实体，箭头是什么类型。"],
      ["split", "760 条用来练，40 条用来小测；小测分数决定保存哪一轮模型。"],
    ],
    table: [
      ["字段", "白话解释", "例子", "模型学什么"],
      ["text", "原始句子", "`... drought tolerance ...`", "从文字里看上下文"],
      ["start/end", "荧光笔坐标", "`start: 10, end: 18`", "实体边界"],
      ["label", "类别贴纸", "`TRT`, `GENE`", "实体或关系类型"],
      ["head/tail", "箭头两端", "基因 -> 性状", "关系方向"],
    ],
    bridge: "NER 学的是 entities 里的圈词答案；RE 学的是 relations 里的箭头答案。",
    quiz: "验证集是给模型继续练习，还是给我们检查它有没有学偏？",
    visual: "data",
  },
  {
    tag: "张量",
    title: "Tensor 是一堆数字排成的积木",
    oneLine: "模型不懂英文，它只懂数字；token、向量、矩阵就是把文字搬进数字世界的方式。",
    deps: [
      "token：把文字切成模型词片",
      "id：每个 token 在词表里的编号",
      "向量：一串数字，像一个词的性格档案",
      "矩阵：很多向量排成表格",
    ],
    steps: [
      ["token", "`drought tolerance` 可能被切成几个词片，像把乐高拆成小块。"],
      ["input_ids", "每个词片换成编号，模型只看编号。"],
      ["embedding", "编号查表变成向量：一串数字描述它的味道、位置、语义。"],
      ["hidden_states", "每一层模型都会更新这些向量，让它们越来越懂上下文。"],
    ],
    table: [
      ["概念", "画面", "形状", "本项目变量"],
      ["token", "纸带上的小格子", "一段文字", "`tokenizer(text)`"],
      ["id", "词典页码", "整数", "`input_ids`"],
      ["vector", "词的性格条形码", "一行数字", "`hidden_size=1024`"],
      ["matrix", "全句子的数字表", "token 数 × 向量长", "`last_hidden_state`"],
    ],
    bridge: "DeBERTa-v3-large 的核心工作，就是把每个 token 的向量不断“调味”，直到能判断它是不是实体。",
    quiz: "模型直接读英文单词，还是先把单词变成数字？",
    visual: "tensor",
  },
  {
    tag: "模型种类",
    title: "模型家族：分类器、CNN、RNN、Transformer、大语言模型",
    oneLine: "它们都是“输入数字，输出判断”的机器，只是观察世界的方式不同。",
    deps: [
      "分类：从几个答案里选一个",
      "序列：顺序很重要的一串东西",
      "上下文：同一个词在不同句子里意思不同",
      "预训练：先读海量材料，再为小任务微调",
    ],
    steps: [
      ["传统分类器", "像只看表格打分的老师，适合简单特征。"],
      ["CNN", "像拿小窗口扫图片，擅长局部图案。"],
      ["RNN", "像从左到右读句子，记忆容易变淡。"],
      ["Transformer", "像全班同学互相传纸条，一次看全句关系。"],
      ["LLM", "更大的 Transformer，目标通常是预测下一个词。"],
    ],
    table: [
      ["模型", "现实类比", "强项", "和你项目的关系"],
      ["Logistic Regression", "打分表", "快、简单", "课程前面乳腺癌分类"],
      ["CNN", "放大镜扫图", "图片局部纹理", "CIFAR-10"],
      ["RNN", "按顺序听故事", "短序列", "class3 字符 RNN"],
      ["Transformer", "全局传纸条", "长文本上下文", "DeBERTa 的底座"],
      ["LLM", "超大文本预测机", "生成和理解", "可做类似任务但成本高"],
    ],
    bridge: "你这次没有训练 ChatGPT 那种生成模型，而是微调一个 Transformer encoder，让它做 NER/RE 判断。",
    quiz: "DeBERTa 更像 CNN，还是 Transformer？",
    visual: "families",
  },
  {
    tag: "Transformer",
    title: "注意力机制：让每个词转头看重要的人",
    oneLine: "Attention（注意力：给别的词分配关注度）像会议室里每个词都在问：我应该听谁的？",
    deps: [
      "一个词的意义取决于附近和远处的词",
      "权重是“我有多在乎你”的数字",
      "层数越多，理解越像多轮讨论",
      "预训练模型已经学过大量通用语言规律",
    ],
    steps: [
      ["会议室画面", "每个 token 是一个人，大家同时互相看。"],
      ["注意力权重", "`gene` 可能更关注 `affects` 和 `drought tolerance`。"],
      ["多层讨论", "第一层看词形，后面看语义、关系、任务线索。"],
      ["微调", "不是从零教英语，而是把通才老师训练成农业信息抽取老师。"],
    ],
    table: [
      ["术语", "白话", "画面", "本项目含义"],
      ["Attention", "看谁更重要", "彩色连线粗细不同", "词之间交换上下文"],
      ["Encoder", "读懂输入的机器", "只读不写作文", "DeBERTa 主体"],
      ["Fine-tuning", "通才转岗培训", "老师学新评分标准", "训练 NER/RE"],
      ["Checkpoint", "阶段性存档", "游戏存档点", "best_ner_model.pt"],
    ],
    bridge: "DeBERTa 的强大来自预训练；你的 800 条数据主要是在告诉它：农业文本里什么叫实体、什么叫关系。",
    quiz: "微调是从零训练语言能力，还是在已有语言能力上教新任务？",
    visual: "attention",
  },
  {
    tag: "NER",
    title: "NER：拿荧光笔给每个 token 贴标签",
    oneLine: "BIOES + CRF 就像一支守规矩的荧光笔，知道实体从哪里开始、在哪里结束。",
    deps: [
      "实体是有边界的文字片段",
      "BIOES 是 token 级边界标签",
      "CRF 是规则感更强的解码器",
      "Precision、Recall、F1 是三种看分数的角度",
    ],
    steps: [
      ["BIOES", "B=开始，I=中间，E=结束，S=单字实体，O=不是实体。"],
      ["分类头", "每个 token 都投票：我是 O，还是 B-GENE，还是 I-TRT？"],
      ["CRF", "像语文老师检查语法：B 后面接 I/E 比乱跳到另一个实体更合理。"],
      ["F1", "既看圈得准不准，也看漏得多不多。"],
    ],
    table: [
      ["指标", "白话", "偏高说明", "偏低说明"],
      ["Precision", "圈出来的有多少是真的", "少误报", "乱圈太多"],
      ["Recall", "真的实体找回多少", "少漏报", "漏掉太多"],
      ["F1", "精确率和召回率折中", "整体平衡", "一边或两边都弱"],
      ["Score", "作业评分公式", "更接近最终分", "需要调参"],
    ],
    bridge: "你看到 NER F1 从 0.17 涨到 0.62，就是荧光笔从乱圈变成比较会圈。",
    quiz: "如果模型只圈了 1 个实体但特别准，Precision 高还是 Recall 高？",
    visual: "ner",
  },
  {
    tag: "RE",
    title: "RE：给两个实体之间画箭头",
    oneLine: "RE 不是先找词，而是在已知两个实体后判断：它们之间有没有关系，关系叫什么。",
    deps: [
      "实体对：两个实体组成一个候选问题",
      "NONE：没有关系也是一种答案",
      "负样本：训练模型学会别乱连线",
      "阈值：决定模型多大胆地输出关系",
    ],
    steps: [
      ["实体对", "如果一句话有 10 个实体，就会产生很多两两组合。"],
      ["Entity Markers", "用 [E1] 和 [E2] 像聚光灯一样标出当前要判断的两个实体。"],
      ["neg_ratio", "负样本比例：告诉模型看多少“没有关系”的例子。"],
      ["threshold", "输出门槛：低了更敢猜，高了更保守。"],
    ],
    table: [
      ["参数", "白话", "太小", "太大"],
      ["neg_ratio", "看多少反例", "容易乱连线", "可能太保守"],
      ["re_threshold", "输出关系门槛", "召回高但误报多", "精确高但漏报多"],
      ["batch_size", "一口吃多少实体对", "慢但稳", "快但可能 OOM"],
      ["epochs", "练几遍实体对题库", "没学够", "可能背题"],
    ],
    bridge: "RE 占总分 60%，所以这次作业里“箭头画得好不好”比“圈词”更影响总分。",
    quiz: "RE 判断的是单个词的标签，还是两个实体之间的箭头？",
    visual: "re",
  },
  {
    tag: "调参",
    title: "调参：不是玄学，是调火候",
    oneLine: "batch_size、lr、epochs、patience 分别像锅大小、火力、翻炒次数、耐心等待次数。",
    deps: [
      "显存限制决定 batch 上限",
      "学习率决定每次改错迈多大步",
      "epoch 决定看几遍训练集",
      "验证集决定何时保存最好的模型",
    ],
    steps: [
      ["batch_size", "锅越大，一次炒越多，更快；锅太大就溢出，也就是 OOM。"],
      ["learning rate", "火力太大容易糊，太小熟得慢；1e-5 是大模型微调的文火。"],
      ["epochs", "多练几遍会变熟，但练太多会只会背答案。"],
      ["patience", "连续几轮没进步就停，避免白烧 GPU。"],
    ],
    table: [
      ["参数", "你现在的值", "为什么这样选", "想冲分怎么动"],
      ["batch_size", "12", "A10 已测可跑，比 8 快", "OOM 就降 8"],
      ["lr", "1e-5", "DeBERTa-large 稳定文火", "loss 抖动就降 5e-6"],
      ["NER epochs", "8", "已够用且会 early stop", "best 在最后一轮再加到 10"],
      ["RE epochs", "6", "先保证完成", "时间够优先加到 8"],
      ["val_ratio", "0.05", "数据少，多留给训练", "想看稳定验证可临时用 0.2"],
    ],
    bridge: "调参不是记公式，而是看症状：OOM 调 batch，学不动看 lr，过拟合看 epochs 和 patience。",
    quiz: "如果 CUDA out of memory，第一反应应该调小哪个参数？",
    visual: "tuning",
  },
];

const enrichments = {
  pipeline: {
    story: [
      "先把论文句子看成一条纸带：模型没有知识，只看到字符和位置。",
      "NER 像荧光笔，负责把纸带上的农业实体圈出来。",
      "RE 像连线题，负责判断两个圈之间有没有语义箭头。",
    ],
    terms: [
      ["Pipeline（流水线）", "把大任务拆成几台机器串起来：先 NER，后 RE。优点是稳、好排查；缺点是前一步错了会影响后一步。"],
      ["Inference（推理）", "训练完成后正式做题。它不再改模型，只用已保存的权重生成答案。"],
      ["Checkpoint（存档）", "训练过程中保存下来的模型文件，像游戏存档；`best_ner_model.pt` 就是目前最好的一次。"],
      ["predictions.json", "最终答卷：每条文本里预测了哪些实体、哪些关系。"],
    ],
    code: [
      ["train_ner.py", "训练“圈词机器”，输出 `output_ner/best_ner_model.pt`。"],
      ["train_re.py", "训练“连线机器”，输出 `output_re/best_re_model.pt`。"],
      ["predict.py", "把两个机器串起来，生成 `predictions.json`。"],
    ],
    pitfalls: [
      "不要以为训练脚本输出的模型就是提交文件；提交/评分通常看的是 `predictions.json`。",
      "不要边训练边删 `output_ner` 或 `output_re`，否则推理找不到权重。",
      "pipeline 方案里 NER 漏掉的实体，RE 后面基本没有机会补回来。",
    ],
  },
  recipe: {
    story: [
      "先把 Python 当成菜谱执行器：一行一行照做，不会自己脑补。",
      "变量是贴了名字的盒子；函数是固定动作的小机器。",
      "命令行参数是外部旋钮，让同一份脚本用不同火候运行。",
    ],
    terms: [
      ["脚本", "`train_re.py` 这种文件就是一份菜谱：从上到下定义工具，最后执行 `main()`。"],
      ["变量", "`args.epochs` 像贴着 epochs 标签的小盒子，里面装数字 6 或 8。"],
      ["函数", "`split_data(...)` 像一台切片机：输入整包数据，输出训练集和验证集。"],
      ["参数", "`--batch_size 12` 是运行时传入的旋钮，不需要改代码本体。"],
    ],
    code: [
      ["parse_args()", "收集你在命令行写的旋钮，例如 batch、epochs、lr。"],
      ["main()", "训练脚本真正开始干活的入口：加载数据、建模型、训练、保存。"],
      ["if __name__ == '__main__'", "只有直接运行这个文件时，才启动训练流程。"],
    ],
    pitfalls: [
      "不要把 `python train_re.py` 理解成打开文件；它是在执行这张菜谱。",
      "命令行里写了参数，会覆盖脚本里的默认值。",
      "同一个脚本可以训练不同配置，区别来自你传入的参数。",
    ],
  },
  data: {
    story: [
      "一条样本像一张阅读理解题：文章、圈词答案、连线答案都在里面。",
      "训练集让模型练习，验证集像小测验，决定哪一轮值得保存。",
      "标签不均衡会让模型偏爱常见类别，所以 RE/NER 的分数要分开看。",
    ],
    terms: [
      ["JSON", "一种规整装数据的文本格式，像很多抽屉嵌套在一起。"],
      ["entity span", "实体坐标：`start/end` 告诉模型荧光笔从哪里划到哪里。"],
      ["relation", "关系箭头：head 是箭头起点，tail 是箭头终点，label 是箭头名字。"],
      ["val_ratio", "切出多少数据当小测验；0.05 表示 5% 验证、95% 训练。"],
    ],
    code: [
      ["load_data()", "读取 `HW_train_data.json`，把 JSON 变成 Python 列表。"],
      ["split_data()", "按 seed 随机打乱，再切成 Train 和 Val。"],
      ["prepare_re_samples()", "把一条文本里的实体两两组合，变成很多 RE 训练题。"],
    ],
    pitfalls: [
      "验证集分数不是最终隐藏测试分，只是我们当前小测验的信号。",
      "val_ratio 太大，训练数据少；太小，验证分数会抖。现在 0.05 是为了多吃训练数据。",
      "数据里少量标注噪声很常见，不代表脚本坏了。",
    ],
  },
  tensor: {
    story: [
      "模型不能直接吃英文，所以先把文本切成 token。",
      "token 再变成 id，id 再查表变成向量。",
      "Transformer 每一层都在更新这些向量，让它们携带更多上下文信息。",
    ],
    terms: [
      ["token", "模型看到的最小文字块，可能是一个词，也可能是半个词。"],
      ["input_ids", "token 在词表里的编号；模型真正吃的是这些数字。"],
      ["embedding", "把编号查表变成一串数字，像词的性格档案。"],
      ["hidden state", "经过模型层层加工后的向量，包含上下文理解。"],
    ],
    code: [
      ["AutoTokenizer.from_pretrained", "加载切词器，把文本切成 token/id。"],
      ["return_offsets_mapping=True", "保留 token 对应原文哪段字符，NER 才能还原 start/end。"],
      ["outputs.last_hidden_state", "模型读完句子后，每个 token 的上下文向量。"],
    ],
    pitfalls: [
      "token 不一定等于英文单词，尤其是基因名、品种名、缩写。",
      "max_length 太短会截断文本，太长会吃显存。",
      "offset_mapping 很关键，丢了它就很难把预测标签还原成字符坐标。",
    ],
  },
  families: {
    story: [
      "所有模型都是把输入数字压成某种判断，只是看世界的镜头不同。",
      "CNN 像滑动放大镜，RNN 像按顺序听故事，Transformer 像全局开会。",
      "LLM 是更大的 Transformer，但你的任务更适合 encoder 微调。",
    ],
    terms: [
      ["Encoder", "只负责读懂输入，不负责写长文；DeBERTa 属于这一类。"],
      ["Decoder", "一个词一个词生成内容，很多聊天模型核心是 decoder。"],
      ["SOTA", "当前效果很强的方案，不等于一定适合明天交作业。"],
      ["Fine-tune", "拿一个已经会语言的模型，教它做你的具体任务。"],
    ],
    code: [
      ["AutoModel", "加载 DeBERTa 这种 encoder 主体。"],
      ["classifier", "在 encoder 顶上接一个小判断头，让它输出标签。"],
      ["CRF / FocalLoss", "任务特化的小部件：NER 需要序列规矩，RE 需要处理不均衡。"],
    ],
    pitfalls: [
      "大模型不等于一定最好；截止时间、显存、数据量都要算进去。",
      "聊天模型擅长生成，但结构化抽取常常用 encoder 更稳。",
      "模型家族不是互斥进化树，而是一堆适合不同任务的工具箱。",
    ],
  },
  attention: {
    story: [
      "每个 token 先带着自己的向量进会议室。",
      "Attention 会给其他 token 分配关注权重：谁更相关，就听谁多一点。",
      "多层 attention 像开很多轮会，最后得到更懂上下文的向量。",
    ],
    terms: [
      ["Attention", "注意力：每个词决定要从其他词那里拿多少信息。"],
      ["Head", "注意力头：一组观察角度；多个头像多个侦探同时看线索。"],
      ["Layer", "模型层：一轮信息交换和加工。DeBERTa-large 有很多层。"],
      ["Pretraining", "预训练：先读海量文本学通用语言规律，再来做小任务。"],
    ],
    code: [
      ["self.encoder(...)", "让所有 token 进入 DeBERTa 的多层注意力网络。"],
      ["attention_mask", "告诉模型哪些位置是真 token，哪些只是 padding。"],
      ["dropout", "训练时随机遮一点信息，防止模型死记硬背。"],
    ],
    pitfalls: [
      "Attention 不是人类意识，它只是数字权重。",
      "预训练模型不是万能知识库；它需要你用标注数据告诉它评分标准。",
      "层数多、参数多会更强，也会更吃显存和时间。",
    ],
  },
  ner: {
    story: [
      "NER 先让每个 token 经过 DeBERTa，得到上下文向量。",
      "分类头给每个 token 打 49 个标签分数：O、B-CROP、I-CROP 等。",
      "CRF 像守门员，挑一串整体最合理的 BIOES 标签。",
    ],
    terms: [
      ["BIOES", "边界标签系统：B 开始、I 中间、O 不是实体、E 结束、S 单独成实体。"],
      ["CRF", "条件随机场：帮序列标签遵守常识，比如 B 后面更常接 I/E。"],
      ["Precision", "圈出来的实体里，有多少是真的。"],
      ["Recall", "所有真的实体里，有多少被找回来了。"],
    ],
    code: [
      ["entities_to_bioes()", "把字符级实体答案变成 BIOES 标签。"],
      ["tokens_to_entities_with_offsets()", "把 token 预测结果还原成 start/end/text/label。"],
      ["best_score", "验证集 score 提升时保存权重，后面变差不会覆盖最佳模型。"],
    ],
    pitfalls: [
      "第 1 轮 F1 低不代表失败，重点看第 2-4 轮是否上升。",
      "验证集只有 40 条，所以 F1 会抖，不要被单轮波动吓到。",
      "NER 漏实体会拖累 RE，因为 RE 只在预测出来的实体上连线。",
    ],
  },
  re: {
    story: [
      "RE 不再从零找词，而是拿两个实体组成一道选择题。",
      "Entity markers 把当前要判断的两个实体用聚光灯标出来。",
      "模型输出 7 类：NONE、CON、USE、HAS、AFF、OCI、LOI。",
    ],
    terms: [
      ["Entity markers", "实体标记：`[E1]...[/E1]` 和 `[E2]...[/E2]`，让模型知道这题问哪两个实体。"],
      ["NONE", "没有关系也是一个类别，模型必须学会不乱连。"],
      ["neg_ratio", "负样本比例：每个正关系配多少个“无关系”例子。"],
      ["threshold", "输出门槛：置信度超过它才把关系写进 predictions.json。"],
    ],
    code: [
      ["insert_entity_markers()", "把 `[E1]`、`[E2]` 插到文本中。"],
      ["prepare_re_samples()", "生成正样本和负样本，形成 RE 训练题库。"],
      ["predict_re_on_entities()", "对 NER 预测出的实体两两判断关系。"],
    ],
    pitfalls: [
      "RE 训练日志里的 pos-micro F1 不是最终总分，只是训练过程的小测。",
      "threshold 太低会乱连，太高会漏连；0.5 是稳妥起点。",
      "RE 样本数比原文本多很多，所以它比 NER 慢是正常的。",
    ],
  },
  tuning: {
    story: [
      "先用 batch_size 找显存上限，别让锅溢出来。",
      "再用 lr 控制每次改错步子，文火更稳。",
      "最后用 epochs 和 patience 控制练习轮数，避免背题。",
    ],
    terms: [
      ["batch_size", "一次喂给显卡多少样本；越大越快，但越吃显存。"],
      ["learning rate", "学习率：每次改错迈多大步。大了会抖，小了会慢。"],
      ["epochs", "训练集看几遍；不是越多越好。"],
      ["patience", "连续几轮没进步就停的耐心值。"],
    ],
    code: [
      ["--batch_size 12", "你已经在 A10 上测过能跑，所以正式版用 12 加速。"],
      ["--val_ratio 0.05", "只留 40 条做小测，剩下 760 条尽量喂给训练。"],
      ["--patience 2", "连续两轮没变好就停，节省 GPU 时间。"],
    ],
    pitfalls: [
      "只看 train loss 会被骗；验证集分数不涨就是危险信号。",
      "重跑会覆盖 output_ner/output_re，跑出好结果先备份。",
      "如果 best epoch 卡在最后一轮，才值得考虑加 epochs。",
    ],
  },
};

const state = {
  current: 0,
  playing: true,
  t: 0,
  labT: 0,
};

const els = {
  lessonList: document.getElementById("lessonList"),
  progressText: document.getElementById("progressText"),
  progressBar: document.getElementById("progressBar"),
  lessonTag: document.getElementById("lessonTag"),
  lessonTitle: document.getElementById("lessonTitle"),
  lessonOneLine: document.getElementById("lessonOneLine"),
  storyRail: document.getElementById("storyRail"),
  dependencyTree: document.getElementById("dependencyTree"),
  bottomUp: document.getElementById("bottomUp"),
  termCards: document.getElementById("termCards"),
  codeLens: document.getElementById("codeLens"),
  pitfallBox: document.getElementById("pitfallBox"),
  compareTable: document.getElementById("compareTable"),
  projectBridge: document.getElementById("projectBridge"),
  tinyQuiz: document.getElementById("tinyQuiz"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  playBtn: document.getElementById("playBtn"),
  canvas: document.getElementById("conceptCanvas"),
  batchSlider: document.getElementById("batchSlider"),
  epochSlider: document.getElementById("epochSlider"),
  lrSlider: document.getElementById("lrSlider"),
  batchValue: document.getElementById("batchValue"),
  epochValue: document.getElementById("epochValue"),
  lrValue: document.getElementById("lrValue"),
  vramMeter: document.getElementById("vramMeter"),
  timeMeter: document.getElementById("timeMeter"),
  overfitMeter: document.getElementById("overfitMeter"),
  vramText: document.getElementById("vramText"),
  timeText: document.getElementById("timeText"),
  overfitText: document.getElementById("overfitText"),
  labAdvice: document.getElementById("labAdvice"),
  trainingCanvas: document.getElementById("trainingCanvas"),
  curveCaption: document.getElementById("curveCaption"),
};

const ctx = els.canvas.getContext("2d");
const trainCtx = els.trainingCanvas.getContext("2d");

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatInline(text) {
  return escapeHtml(text).replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderNav() {
  els.lessonList.innerHTML = lessons.map((lesson, index) => `
    <button class="lesson-button ${index === state.current ? "active" : ""}" type="button" data-index="${index}">
      <span class="lesson-number">${index + 1}</span>
      <span>
        <strong>${escapeHtml(lesson.title)}</strong>
        <span>${escapeHtml(lesson.tag)}</span>
      </span>
    </button>
  `).join("");

  els.lessonList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => setLesson(Number(button.dataset.index)));
  });
}

function renderLesson() {
  const lesson = lessons[state.current];
  const extra = enrichments[lesson.visual];
  els.lessonTag.textContent = lesson.tag;
  els.lessonTitle.textContent = lesson.title;
  els.lessonOneLine.textContent = lesson.oneLine;
  els.progressText.textContent = `${state.current + 1} / ${lessons.length}`;
  els.progressBar.style.width = `${((state.current + 1) / lessons.length) * 100}%`;

  els.storyRail.innerHTML = extra.story.map((item, index) => `
    <div class="story-dot" style="--delay:${index * 0.35}s">
      <b>${index + 1}</b>
      <span>${formatInline(item)}</span>
    </div>
  `).join("");

  els.dependencyTree.innerHTML = lesson.deps.map((item) => `<li>${formatInline(item)}</li>`).join("");
  els.bottomUp.innerHTML = lesson.steps.map(([title, body]) => `
    <div class="step-item">
      <strong>${escapeHtml(title)}</strong>
      <span>${formatInline(body)}</span>
    </div>
  `).join("");

  els.termCards.innerHTML = extra.terms.map(([term, body]) => `
    <div class="term-card">
      <strong>${formatInline(term)}</strong>
      <p>${formatInline(body)}</p>
    </div>
  `).join("");

  els.codeLens.innerHTML = extra.code.map(([code, body]) => `
    <div class="code-line">
      <code>${escapeHtml(code)}</code>
      <span>${formatInline(body)}</span>
    </div>
  `).join("");

  els.pitfallBox.innerHTML = extra.pitfalls.map((item, index) => `
    <div class="pitfall-item">
      <b>${index + 1}</b>
      <span>${formatInline(item)}</span>
    </div>
  `).join("");

  const [head, ...rows] = lesson.table;
  els.compareTable.innerHTML = `
    <table>
      <thead><tr>${head.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${formatInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;

  els.projectBridge.innerHTML = formatInline(lesson.bridge);
  els.tinyQuiz.innerHTML = `
    <strong>小问题：</strong>${escapeHtml(lesson.quiz)}
    <details>
      <summary>点我看提示</summary>
      <p>别背答案，只问自己：这个东西在流水线里负责“圈词”、还是负责“连线”、还是负责“控制火候”？</p>
    </details>
  `;

  renderNav();
}

function setLesson(index) {
  state.current = (index + lessons.length) % lessons.length;
  state.t = 0;
  renderLesson();
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  els.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const trainRect = els.trainingCanvas.getBoundingClientRect();
  els.trainingCanvas.width = Math.max(1, Math.floor(trainRect.width * dpr));
  els.trainingCanvas.height = Math.max(1, Math.floor(trainRect.height * dpr));
  trainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function roundedRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawLabel(text, x, y, color = "#dff3ed") {
  ctx.fillStyle = color;
  ctx.font = "700 14px ui-sans-serif";
  ctx.fillText(text, x, y);
}

function drawToken(text, x, y, color = "#ffffff", fill = "#20313c") {
  ctx.fillStyle = fill;
  roundedRect(x, y, 112, 42, 8);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = "700 13px ui-sans-serif";
  ctx.fillText(text, x + 12, y + 26);
}

function drawArrow(x1, y1, x2, y2, color = "#7fd0bd") {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 10 * Math.cos(angle - 0.45), y2 - 10 * Math.sin(angle - 0.45));
  ctx.lineTo(x2 - 10 * Math.cos(angle + 0.45), y2 - 10 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawBackground(w, h) {
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 42) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 42) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  for (let i = 0; i < 24; i++) {
    const x = (i * 97 + state.t * (0.35 + (i % 4) * 0.11)) % w;
    const y = (i * 53 + Math.sin(state.t * 0.03 + i) * 16 + h) % h;
    ctx.fillStyle = `rgba(127, 208, 189, ${0.08 + (i % 5) * 0.025})`;
    ctx.beginPath();
    ctx.arc(x, y, 1.8 + (i % 3), 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawMovingDot(x1, y1, x2, y2, phase, color = "#fff0cc") {
  const p = (phase % 1 + 1) % 1;
  const x = x1 + (x2 - x1) * p;
  const y = y1 + (y2 - y1) * p;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();
}

function drawScene() {
  const w = els.canvas.clientWidth;
  const h = els.canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  drawBackground(w, h);

  const lesson = lessons[state.current];
  const t = state.t;
  const pulse = (Math.sin(t * 0.05) + 1) / 2;

  ctx.save();
  ctx.translate(Math.max(16, w * 0.04), Math.max(20, h * 0.08));

  if (lesson.visual === "pipeline") drawPipeline(w, h, pulse);
  else if (lesson.visual === "recipe") drawRecipe(w, h, pulse);
  else if (lesson.visual === "data") drawData(w, h, pulse);
  else if (lesson.visual === "tensor") drawTensor(w, h, pulse);
  else if (lesson.visual === "families") drawFamilies(w, h, pulse);
  else if (lesson.visual === "attention") drawAttention(w, h, pulse);
  else if (lesson.visual === "ner") drawNer(w, h, pulse);
  else if (lesson.visual === "re") drawRe(w, h, pulse);
  else drawTuning(w, h, pulse);

  ctx.restore();

  drawTrainingCurve();

  if (state.playing) {
    state.t += 1;
    state.labT += 1;
  }
  requestAnimationFrame(drawScene);
}

function drawPipeline(w, h, pulse) {
  const y = h * 0.38;
  drawToken("text", 10, y);
  drawToken("NER", 170, y, "#dff3ed", "#173b35");
  drawToken("entities", 330, y);
  drawToken("RE", 490, y, "#e2ebff", "#182b52");
  drawToken("relations", 650, y);
  [132, 292, 452, 612].forEach((x) => drawArrow(x, y + 21, x + 36, y + 21));
  [132, 292, 452, 612].forEach((x, i) => drawMovingDot(x, y + 21, x + 36, y + 21, state.t * 0.025 + i * 0.2));
  drawLabel("先圈词，再连线，最后生成 predictions.json", 20, y - 42);
  ctx.globalAlpha = 0.35 + pulse * 0.55;
  drawArrow(695, y + 68, 695, y + 122, "#f0c36a");
  ctx.globalAlpha = 1;
  drawToken("predictions.json", 620, y + 132, "#fff0cc", "#4a3715");
}

function drawRecipe(w, h, pulse) {
  drawLabel("脚本像菜谱：按步骤处理原料", 20, 34);
  const steps = ["load_data", "build_model", "train", "save"];
  steps.forEach((s, i) => {
    drawToken(s, 35 + i * 150, 120 + Math.sin((state.t + i * 16) * 0.05) * 8);
    if (i < steps.length - 1) {
      drawArrow(148 + i * 150, 142, 182 + i * 150, 142);
      drawMovingDot(148 + i * 150, 142, 182 + i * 150, 142, state.t * 0.03 + i * 0.25);
    }
  });
  ctx.fillStyle = `rgba(127, 208, 189, ${0.2 + pulse * 0.45})`;
  roundedRect(45, 230, 580, 56, 10);
  ctx.fill();
  drawLabel("--batch_size 12 是旋钮，不是魔法", 70, 264, "#ffffff");
}

function drawData(w, h, pulse) {
  drawLabel("一条样本 = text + entities + relations", 20, 34);
  drawToken("text", 40, 95, "#fff", "#29333d");
  drawToken("entities", 230, 95, "#dff3ed", "#173b35");
  drawToken("relations", 420, 95, "#e2ebff", "#182b52");
  drawArrow(152, 116, 228, 116);
  drawArrow(342, 116, 418, 116);
  drawMovingDot(152, 116, 228, 116, state.t * 0.025);
  drawMovingDot(342, 116, 418, 116, state.t * 0.025 + 0.4);
  const x = 80;
  const y = 220;
  ["GENE", "AFF", "TRT"].forEach((s, i) => {
    const fill = i === 1 ? "#4a3715" : "#173b35";
    drawToken(s, x + i * 160, y + Math.sin(state.t * 0.04 + i) * 8, "#fff", fill);
  });
  drawArrow(x + 112, y + 21, x + 160, y + 21, "#f0c36a");
  drawArrow(x + 272, y + 21, x + 320, y + 21, "#f0c36a");
  drawMovingDot(x + 112, y + 21, x + 160, y + 21, state.t * 0.03, "#fff0cc");
  drawMovingDot(x + 272, y + 21, x + 320, y + 21, state.t * 0.03 + 0.5, "#fff0cc");
}

function drawTensor(w, h, pulse) {
  drawLabel("文字被切成 token，再变成数字矩阵", 20, 34);
  const words = ["drought", "tolerance", "gene", "QTL"];
  words.forEach((word, i) => drawToken(word, 20 + i * 135, 80));
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 12; col++) {
      const val = (Math.sin(state.t * 0.04 + row * 0.8 + col * 0.4) + 1) / 2;
      ctx.fillStyle = `rgba(127, 208, 189, ${0.18 + val * 0.75})`;
      ctx.fillRect(50 + col * 24, 190 + row * 32, 18, 24);
    }
  }
  drawLabel("embedding / hidden states", 360, 245, "#e2ebff");
}

function drawFamilies(w, h, pulse) {
  const items = [
    ["MLP", "#334155"],
    ["CNN", "#0f7b68"],
    ["RNN", "#b47b10"],
    ["Transformer", "#315a9f"],
    ["LLM", "#c4523b"],
  ];
  drawLabel("模型家族都是“数字输入 -> 判断输出”", 20, 34);
  items.forEach(([name, fill], i) => {
    const radius = 34 + i * 8 + pulse * 6;
    ctx.beginPath();
    ctx.arc(80 + i * 145, 190, radius, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "800 15px ui-sans-serif";
    ctx.fillText(name, 52 + i * 145, 196);
  });
}

function drawAttention(w, h, pulse) {
  drawLabel("Attention：每个词都转头看关键线索", 20, 34);
  const tokens = [
    ["gene", 90, 105],
    ["affects", 305, 70],
    ["drought", 520, 140],
    ["tolerance", 300, 250],
  ];
  tokens.forEach(([text, x, y]) => drawToken(text, x, y));
  const center = tokens[0];
  tokens.slice(1).forEach(([, x, y], i) => {
    ctx.globalAlpha = 0.35 + pulse * (0.2 + i * 0.18);
    drawArrow(center[1] + 112, center[2] + 21, x, y + 21, ["#f0c36a", "#7fd0bd", "#9ab6ff"][i]);
    drawMovingDot(center[1] + 112, center[2] + 21, x, y + 21, state.t * (0.015 + i * 0.006), "#ffffff");
  });
  ctx.globalAlpha = 1;
}

function drawNer(w, h, pulse) {
  drawLabel("NER：给每个 token 贴 BIOES 标签", 20, 34);
  const tokens = [["sorghum", "S-CROP"], ["drought", "B-TRT"], ["tolerance", "E-TRT"], ["gene", "S-GENE"]];
  tokens.forEach(([word, tag], i) => {
    const x = 30 + i * 160;
    drawToken(word, x, 120);
    ctx.fillStyle = i === 1 || i === 2 ? "#fff0cc" : "#dff3ed";
    roundedRect(x + 8, 176 + pulse * 8, 96, 30, 7);
    ctx.fill();
    ctx.fillStyle = "#172026";
    ctx.font = "800 13px ui-sans-serif";
    ctx.fillText(tag, x + 18, 196 + pulse * 8);
  });
}

function drawRe(w, h, pulse) {
  drawLabel("RE：聚光灯照两个实体，判断箭头类型", 20, 34);
  drawToken("[E1] gene", 90, 120, "#dff3ed", "#173b35");
  drawToken("[E2] trait", 470, 120, "#e2ebff", "#182b52");
  drawArrow(205, 140, 468, 140, "#f0c36a");
  drawMovingDot(205, 140, 468, 140, state.t * 0.018, "#fff0cc");
  ctx.fillStyle = `rgba(255, 240, 204, ${0.25 + pulse * 0.55})`;
  roundedRect(275, 180, 160, 52, 10);
  ctx.fill();
  drawLabel("AFF / HAS / LOI / NONE", 292, 212, "#ffffff");
}

function drawTuning(w, h, pulse) {
  drawLabel("调参像调火候：锅、火、次数、耐心", 20, 34);
  const knobs = [
    ["batch", 95, "#0f7b68"],
    ["lr", 255, "#b47b10"],
    ["epochs", 415, "#315a9f"],
    ["patience", 575, "#c4523b"],
  ];
  knobs.forEach(([name, x, color], i) => {
    ctx.beginPath();
    ctx.arc(x, 170, 48, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x, 170);
    const a = -Math.PI / 2 + pulse * Math.PI * 1.5 + i * 0.2;
    ctx.lineTo(x + Math.cos(a) * 36, 170 + Math.sin(a) * 36);
    ctx.stroke();
    drawLabel(name, x - 28, 245, "#ffffff");
  });
}

function drawTrainingCurve() {
  const w = els.trainingCanvas.clientWidth;
  const h = els.trainingCanvas.clientHeight;
  const batch = Number(els.batchSlider.value);
  const epochs = Number(els.epochSlider.value);
  const lrRaw = Number(els.lrSlider.value);
  const lrFactor = lrRaw / 10;
  const speed = Math.min(1.4, batch / 12);
  const risk = Math.max(0, (epochs - 7) / 12) + Math.max(0, (lrRaw - 12) / 30);
  const t = state.labT * 0.015 * speed;

  trainCtx.clearRect(0, 0, w, h);
  trainCtx.fillStyle = "#101820";
  trainCtx.fillRect(0, 0, w, h);

  trainCtx.strokeStyle = "rgba(255,255,255,0.08)";
  trainCtx.lineWidth = 1;
  for (let x = 34; x < w; x += 46) {
    trainCtx.beginPath();
    trainCtx.moveTo(x, 18);
    trainCtx.lineTo(x, h - 24);
    trainCtx.stroke();
  }
  for (let y = 28; y < h - 18; y += 34) {
    trainCtx.beginPath();
    trainCtx.moveTo(28, y);
    trainCtx.lineTo(w - 18, y);
    trainCtx.stroke();
  }

  function curvePoint(i, kind) {
    const x = 34 + i * ((w - 70) / 80);
    const progress = i / 80;
    const wiggle = Math.sin(i * 0.35 + state.labT * 0.05) * 0.035;
    if (kind === "loss") {
      const y = 34 + (h - 78) * Math.exp(-progress * (2.4 + lrFactor * 0.22)) + wiggle * h;
      return [x, Math.min(h - 28, Math.max(24, y))];
    }
    const rise = 1 - Math.exp(-progress * (2.1 + lrFactor * 0.18));
    const overfitDip = Math.max(0, progress - 0.62) * risk * 0.42;
    const y = h - 34 - (h - 76) * (rise - overfitDip) + wiggle * h;
    return [x, Math.min(h - 28, Math.max(24, y))];
  }

  function drawCurve(kind, color) {
    trainCtx.strokeStyle = color;
    trainCtx.lineWidth = 3;
    trainCtx.beginPath();
    for (let i = 0; i <= 80; i++) {
      const [x, y] = curvePoint(i, kind);
      if (i === 0) trainCtx.moveTo(x, y);
      else trainCtx.lineTo(x, y);
    }
    trainCtx.stroke();
  }

  drawCurve("loss", "#f0c36a");
  drawCurve("score", "#7fd0bd");

  const cursor = Math.floor((t % 1) * 80);
  const [lx, ly] = curvePoint(cursor, "loss");
  const [sx, sy] = curvePoint(cursor, "score");
  trainCtx.fillStyle = "#f0c36a";
  trainCtx.beginPath();
  trainCtx.arc(lx, ly, 5, 0, Math.PI * 2);
  trainCtx.fill();
  trainCtx.fillStyle = "#7fd0bd";
  trainCtx.beginPath();
  trainCtx.arc(sx, sy, 5, 0, Math.PI * 2);
  trainCtx.fill();

  trainCtx.fillStyle = "#d8d6c8";
  trainCtx.font = "700 13px ui-sans-serif";
  trainCtx.fillText("loss 下降", 34, 20);
  trainCtx.fillText("score 上升", w - 120, 20);
  els.curveCaption.textContent = `动画含义：batch=${batch} 时训练推进更快；epochs=${epochs} 越多，后段过拟合风险越明显；lr=${(lrRaw / 10).toFixed(1)}e-5 控制曲线抖动和学习速度。`;
}

function updateLab() {
  const batch = Number(els.batchSlider.value);
  const epochs = Number(els.epochSlider.value);
  const lrRaw = Number(els.lrSlider.value);
  const lr = lrRaw / 10;
  const vram = Math.min(100, Math.round((batch / 24) * 100));
  const time = Math.min(100, Math.round((epochs * 9) * (12 / batch)));
  const overfit = Math.min(100, Math.round(Math.max(0, epochs - 5) * 9 + Math.max(0, lrRaw - 12) * 2));

  els.batchValue.textContent = String(batch);
  els.epochValue.textContent = String(epochs);
  els.lrValue.textContent = `${lr.toFixed(1)}e-5`;
  els.vramMeter.style.width = `${vram}%`;
  els.timeMeter.style.width = `${time}%`;
  els.overfitMeter.style.width = `${overfit}%`;
  els.vramText.textContent = vram > 80 ? "危险：可能 OOM" : vram > 55 ? "偏高但可试" : "稳";
  els.timeText.textContent = time > 85 ? "偏久" : time > 55 ? "中等" : "快";
  els.overfitText.textContent = overfit > 65 ? "偏高" : overfit > 35 ? "可观察" : "低";

  const advice = [];
  if (batch > 12) advice.push("batch 超过你已验证的 12，正式跑前先测 1 epoch。");
  else advice.push("batch 在已验证范围内，主要看速度和显存。");
  if (epochs > 8) advice.push("epoch 增加可能涨分，也可能过拟合；看 best epoch 是否卡在最后一轮。");
  if (lrRaw > 15) advice.push("学习率偏大，loss 抖动时先降到 1e-5 或 5e-6。");
  if (lrRaw < 6) advice.push("学习率很小，稳但可能学得慢。");
  els.labAdvice.innerHTML = advice.map((x) => `<p>${escapeHtml(x)}</p>`).join("");
  drawTrainingCurve();
}

els.prevBtn.addEventListener("click", () => setLesson(state.current - 1));
els.nextBtn.addEventListener("click", () => setLesson(state.current + 1));
els.playBtn.addEventListener("click", () => {
  state.playing = !state.playing;
  els.playBtn.textContent = state.playing ? "暂停动画" : "继续动画";
});

[els.batchSlider, els.epochSlider, els.lrSlider].forEach((input) => {
  input.addEventListener("input", updateLab);
});

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
renderLesson();
updateLab();
requestAnimationFrame(drawScene);
