# data_utils.py 数据处理可视化

这是一个零依赖静态网页，用来解释 `FinalTestv2/data_utils.py` 如何把原始 JSON 数据处理成 NER 和 RE 训练样本。

## 打开方式

最简单方式：

```bash
cd /Users/ray/Documents/Fork/DeepLearning/FinalTestv2/data_processing_visualizer
/Users/ray/miniconda3/bin/python -m http.server 8765 --bind 127.0.0.1
```

然后浏览器打开：

```text
http://127.0.0.1:8765/
```

也可以直接双击 `index.html`，但部分浏览器对本地文件限制更严格，用上面的本地服务器更稳。

## 页面内容

- 总览：展示 JSON 如何分成 NERDataset、REDataset、Metrics 三条主线。
- 标签体系：展示 12 类实体如何配合 BIOES 生成 49 个 NER 标签，6 类关系如何加上 NONE 生成 7 个 RE 标签。
- NER：展示实体 span -> 字符 BIOES -> token offset 对齐 -> tensor -> 预测后还原实体。
- RE：展示实体对枚举、正负样本、neg_ratio、Entity Markers、REDataset 输出。
- 评分：展示 TP/FP/FN、Precision、Recall、F1、Score 和最终总分公式。

说明：网页里的 tokenizer 是轻量演示版，用来解释 `offset_mapping` 的思想；真实训练仍然使用 `transformers` 里的 DeBERTa tokenizer。
