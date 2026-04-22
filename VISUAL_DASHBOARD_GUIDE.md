# 可视化训练面板使用说明（Streamlit）

可视化应用文件：
- [visual_train_dashboard.py](/Users/ray/Documents/Fork/DeepLearning/visual_train_dashboard.py)

---

## 1. 启动方式

```bash
conda run -n ml_env streamlit run /Users/ray/Documents/Fork/DeepLearning/visual_train_dashboard.py
```

启动后在浏览器打开终端提示的本地地址（通常是 `http://localhost:8501`）。

---

## 2. 这个面板能看到什么

- 中英双语 UI（中文 / English）
- 数据样本区：
  - 显示 MNIST 输入图像和标签
- 网络结构区：
  - 可视化三层结构（输入层 -> 隐藏层 -> 输出层）
- 训练过程区（实时）：
  - 进度条（按 epoch/batch 更新）
  - 当前 batch 的 loss / 运行中训练准确率
  - Loss 曲线（train_loss + val_loss）
  - 准确率曲线（train_acc + val_acc）
  - 每个 epoch 的指标表
  - 混淆矩阵（测试集）
- 结果区：
  - 最佳验证准确率
  - 最佳模型保存路径

---

## 3. 可视化调参（左侧栏）

可直接调：
- `hidden_dim`：隐藏层宽度
- `dropout`：随机失活比例
- `learning_rate`
- `weight_decay`
- `batch_size`
- `epochs`
- `max batches/epoch`（演示时可限速）
- `optimizer`（AdamW/SGD）
- `data_dir`、`save_path`

还有预设：
- `Balanced`
- `Fast`
- `Regularized`

---

## 4. 建议调参顺序

1. 先用 `Balanced` 跑通  
2. 若训练快但准确率低：增大 `hidden_dim` 或 `epochs`  
3. 若训练高、验证低（过拟合）：提高 `dropout` / `weight_decay`  
4. 若收敛慢：小幅提高 `learning_rate`  
5. 若震荡大：降低 `learning_rate` 或增大 `batch_size`

---

## 5. 最终模型和验证

- 最佳模型会保存到 `save_path`（默认 `best_mlp_mnist.pth`）。
- 验证依据：
  - `val_acc`（测试集准确率）
  - 混淆矩阵（看哪些类别容易混淆）

如果你愿意，下一步我可以给这个面板再加两项：
- “上传手写图片进行预测”
- “训练结束自动导出实验报告（CSV + PNG 图）”
