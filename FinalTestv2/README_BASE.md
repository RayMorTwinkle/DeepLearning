# FinalTestv2 Base 实验线

这是小模型备用方案，不替代 `FinalTestCloud` 的 large 主线。

## 文件

| 文件 | 用途 |
|------|------|
| `main_base.ipynb` | 单文件 Notebook，内嵌训练/推理脚本 |
| `HW_train_data.json` | 训练数据 |
| `gen_main_base.py` | 本地重新生成 Notebook 用 |

## 云端怎么跑

上传到 Notebook 工作目录：

```text
main_base.ipynb
HW_train_data.json
HW_test_data.json      # 如果这是带 entities/relations 的验证集，也一起上传
```

在 Colab 里，数据文件放在根目录或 `sample_data/` 都可以，Notebook 会自动找。

如果 `HW_test_data.json` 带有 `entities` 和 `relations`，Notebook 会把它作为固定验证集和输出集。  
它不会被合并进训练集。

然后按顺序运行 Cell：

1. 环境与依赖
2. 数据检查
3. 写出运行脚本
4. 下载 `microsoft/deberta-v3-base`
5. 只用 `HW_train_data.json` 训练 NER base，用 `HW_test_data.json` 验证
6. 只用 `HW_train_data.json` 训练 RE base，用 `HW_test_data.json` 验证
7. 在 `HW_test_data.json` 上扫 RE 阈值并生成 `predictions_base.json`

Cell 8 默认会读取 `HW_test_data.json` 并输出：

```text
predictions_test.json
```

如果 `HW_test_data.json` 里有 `entities`，Cell 8 默认使用这些实体，只预测关系。  
如果想模拟真正测试集流程，把 Cell 8 里的 `USE_PROVIDED_ENTITIES = True` 改成 `False`。

## 默认参数

| 模块 | 参数 |
|------|------|
| NER | `epochs=10`, `batch_size=24`, `lr=2e-5`, `dropout=0.15`, `patience=3` |
| RE | `epochs=10`, `batch_size=24`, `lr=2e-5`, `dropout=0.15`, `neg_ratio=1`, `patience=3` |
| 推理 | 自动扫 `re_threshold` |

如果 A10 出现 OOM，把 Notebook 里两个训练 Cell 的 `--batch_size 24` 改成 `16`。

Colab 默认直接下载 HuggingFace，不使用 `hf-mirror`。如果下载失败，先重启会话再重跑 Cell 1 和 Cell 4。

## 预期

Base 模型更快、更省显存，但不保证比分数已经跑通的 large 更高。  
它的价值是快速多试一条路线，看 pipeline 的 NER 和 RE 是否更稳。
