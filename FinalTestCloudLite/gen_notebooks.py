"""
Notebook 生成脚本
用法: conda run -n ml_env python gen_notebooks.py

修改此文件后重新运行，即可重新生成 main.ipynb 和 main_lite.ipynb
"""

import os
import json

try:
    import nbformat as nbf
except ModuleNotFoundError:
    class _Notebook(dict):
        @property
        def cells(self):
            return self["cells"]

        @cells.setter
        def cells(self, value):
            self["cells"] = value

    class _MiniNotebookFormat:
        class v4:
            @staticmethod
            def new_notebook():
                return _Notebook({
                    "cells": [],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                })

            @staticmethod
            def new_markdown_cell(source):
                return {"cell_type": "markdown", "metadata": {}, "source": source}

            @staticmethod
            def new_code_cell(source):
                return {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }

        @staticmethod
        def write(nb, path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, ensure_ascii=False, indent=2)
                f.write("\n")

    nbf = _MiniNotebookFormat()

# ============== 通用配置 ==============
HF_ENDPOINT = "https://hf-mirror.com"
MODEL_NAME = "microsoft/deberta-v3-large"
LOCAL_MODEL_DIR = "./deberta-v3-large"
DATA_FILE = "HW_train_data.json"


def make_cell_env_check():
    """Cell 1: 环境检查"""
    code = f'''\
# 设置 HuggingFace 镜像源（国内平台必须）
import os
os.environ['HF_ENDPOINT'] = '{HF_ENDPOINT}'

# 清理可能存在的坏缓存
import shutil
for d in ['/root/.cache/huggingface/hub/models--microsoft--deberta-v3-large']:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f'Cleared bad cache: {{d}}')

import torch, sys

print("=" * 50)
print(" Environment")
print("=" * 50)
print(f"PyTorch: {{torch.__version__}}")
print(f"CUDA:    {{torch.cuda.is_available()}}")

if torch.cuda.is_available():
    print(f"GPU:     {{torch.cuda.get_device_name(0)}}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory:  {{mem:.1f}} GB")
else:
    print("WARNING: CUDA is False. Do not start training until the runtime uses a GPU PyTorch build.")

print(f"Files:   {{sorted(os.listdir('.'))}}")
print("=" * 50)
'''
    return nbf.v4.new_markdown_cell("### Cell 1: 环境检查\n\n**预期输出**: CUDA=True, GPU信息, 文件列表"), nbf.v4.new_code_cell(code)


def make_cell_install_deps():
    """Cell 2: 安装依赖"""
    md = "### Cell 2: 安装依赖\n\n安装 transformers, sentencepiece, pytorch-crf, scikit-learn, tqdm"
    code = '''\
!pip install "transformers<5" sentencepiece protobuf pytorch-crf scikit-learn tqdm -q
import torch, transformers
print(f"torch {torch.__version__}, transformers {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Dependencies OK")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_cell_data_check():
    """Cell 3: 数据检查"""
    md = "### Cell 3: 数据快速检查\n\n**预期输出**: 800 样本, 实体/关系统计"
    code = f'''\
import json
from collections import Counter

data = json.load(open("{DATA_FILE}"))
print(f"样本数: {{len(data)}}")

ent_counter = Counter()
for d in data:
    for e in d["entities"]:
        ent_counter[e["label"]] += 1
print(f"实体类型: {{len(ent_counter)}}, 总实体: {{sum(ent_counter.values())}}")

rel_counter = Counter()
for d in data:
    for r in d["relations"]:
        rel_counter[r["label"]] += 1
print(f"关系类型: {{len(rel_counter)}}, 总关系: {{sum(rel_counter.values())}}")
print(f"\\n数据 OK")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_cell_download_model():
    """Cell 4: 预下载模型"""
    md = "### Cell 4: 预下载模型\n\n把模型下载到可见文件夹，只下一次，NER和RE共用。\n\n**预计耗时**: ~6-8 分钟"
    code = f'''\
import os
os.environ['HF_ENDPOINT'] = '{HF_ENDPOINT}'

from huggingface_hub import snapshot_download

print("Downloading {MODEL_NAME}...")
snapshot_download(
    "{MODEL_NAME}",
    local_dir="{LOCAL_MODEL_DIR}",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5"],
)
print("✅ Downloaded to {LOCAL_MODEL_DIR}")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_cell_train_ner(epochs, patience, batch_size=8, lr="1e-5", val_ratio=0.2):
    """NER 训练 Cell
    A10 24GB 已测 batch=12 可用；Lite 默认 8，Full 默认 12。
    """
    md = f"""### Cell 5: 训练 NER ({epochs} epochs)

使用 DeBERTa-v3-large + CRF 做序列标注。

| 参数 | 值 |
|------|-----|
| model | {LOCAL_MODEL_DIR} |
| epochs | {epochs} |
| batch_size | {batch_size} |
| lr | {lr} |
| patience | {patience} |
| val_ratio | {val_ratio} |
"""
    code = f'''\
import torch
assert torch.cuda.is_available(), "CUDA=False：当前不是 GPU PyTorch 环境，请先切换/修复运行时再训练。"

!python train_ner.py \\
    --data {DATA_FILE} \\
    --model {LOCAL_MODEL_DIR} \\
    --save_dir ./output_ner \\
    --max_length 384 \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --lr {lr} \\
    --warmup_ratio 0.1 \\
    --dropout 0.1 \\
    --val_ratio {val_ratio} \\
    --patience {patience}

import os
assert os.path.exists("./output_ner/best_ner_model.pt"), "NER model not saved!"
print(f"NER model: {{os.path.getsize('./output_ner/best_ner_model.pt')/1e9:.2f}} GB")
print("NER PASSED")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_cell_train_re(epochs, neg_ratio, patience, batch_size=8, lr="1e-5", val_ratio=0.2):
    """RE 训练 Cell
    A10 24GB 已测 batch=12 可用；Lite 默认 8，Full 默认 12。
    neg_ratio 控制负样本比例，1=正负1:1
    """
    md = f"""### Cell 6: 训练 RE ({epochs} epochs, neg_ratio={neg_ratio})

使用 DeBERTa-v3-large + Entity Markers 做关系分类。

| 参数 | 值 |
|------|-----|
| model | {LOCAL_MODEL_DIR} |
| epochs | {epochs} |
| batch_size | {batch_size} |
| lr | {lr} |
| neg_ratio | {neg_ratio} |
| patience | {patience} |
| val_ratio | {val_ratio} |
"""
    code = f'''\
import torch
assert torch.cuda.is_available(), "CUDA=False：当前不是 GPU PyTorch 环境，请先切换/修复运行时再训练。"

!python train_re.py \\
    --data {DATA_FILE} \\
    --model {LOCAL_MODEL_DIR} \\
    --save_dir ./output_re \\
    --max_length 384 \\
    --batch_size {batch_size} \\
    --epochs {epochs} \\
    --lr {lr} \\
    --warmup_ratio 0.1 \\
    --dropout 0.1 \\
    --neg_ratio {neg_ratio} \\
    --val_ratio {val_ratio} \\
    --patience {patience}

import os
assert os.path.exists("./output_re/best_re_model.pt"), "RE model not saved!"
print(f"RE model: {{os.path.getsize('./output_re/best_re_model.pt')/1e9:.2f}} GB")
print("RE PASSED")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_cell_predict(val_ratio=0.2, re_threshold=0.9):
    """推理 Cell"""
    md = """### Cell 7: 联合推理 + 评估

加载 NER + RE 模型，跑完整 Pipeline。

**预期输出**: predictions.json + P/R/F1/Score
"""
    code = f'''\
!python predict.py \\
    --ner_model_dir ./output_ner \\
    --re_model_dir ./output_re \\
    --data {DATA_FILE} \\
    --output predictions.json \\
    --batch_size 12 \\
    --re_threshold {re_threshold} \\
    --max_length 384 \\
    --val_ratio {val_ratio} \\
    --evaluate

import os, json
assert os.path.exists("predictions.json"), "predictions.json not found!"

with open("predictions.json") as f:
    preds = json.load(f)
total_ent = sum(len(p['entities']) for p in preds)
total_rel = sum(len(p['relations']) for p in preds)
print(f"\\n预测: {{len(preds)}} 文本, {{total_ent}} 实体, {{total_rel}} 关系")
print("Inference PASSED")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_lite_notebook():
    """生成 Lite 版 Notebook"""
    nb = nbf.v4.new_notebook()
    cells = []

    # 标题
    cells.append(nbf.v4.new_markdown_cell("""# FinalTest Lite — 快速验证版

> ⚠️ **这是验证版，只用来排除bug，分数不代表最终结果！**
> 完整训练请用 `FinalTestCloud/main.ipynb`

| 项目 | Lite版 | 完整版 |
|------|--------|--------|
| NER epochs | **2** | 8 |
| RE epochs | **2** | 6 |
| RE neg_ratio | **1** | 1 |
| 预计耗时 | **~5-6 分钟** | ~45 分钟 |

按顺序执行每个 Cell，全部通过后就安全了。
"""))

    # 标准 Cells
    cells.extend(make_cell_env_check())
    cells.extend(make_cell_install_deps())
    cells.extend(make_cell_data_check())
    cells.extend(make_cell_download_model())
    cells.extend(make_cell_train_ner(epochs=2, patience=2))
    cells.extend(make_cell_train_re(epochs=2, neg_ratio=1, patience=2))
    cells.extend(make_cell_predict())

    # 结尾
    cells.append(nbf.v4.new_markdown_cell("""### ✅ Lite 验证通过！

所有模块运行正常。现在可以运行完整版 `main.ipynb` 了。

---

#### 排查指南

| 报错位置 | 常见原因 | 解决 |
|----------|----------|------|
| Cell 2 pip install | 网络问题 | 重试 |
| Cell 4 下载模型 | HuggingFace 被墙 | 已内置 HF_ENDPOINT |
| Cell 5 NER CUDA OOM | 显存不够 | 改 `--batch_size 4` |
| Cell 6 RE CUDA OOM | 显存不够 | 改 `--batch_size 4` |
| Cell 5/6 loss=nan | 学习率太大 | 改 `--lr 5e-6` |
"""))

    nb.cells = cells
    return nb


def make_full_notebook():
    """生成完整版 Notebook"""
    nb = nbf.v4.new_notebook()
    cells = []

    # 标题
    cells.append(nbf.v4.new_markdown_cell("""# FinalTest: 联合命名实体识别 + 关系抽取

> **杂粮作物领域 · NER(12类实体) + RE(6类关系)**

| 项目 | 说明 |
|------|------|
| 模型 | DeBERTa-v3-large + CRF (NER) / Entity Markers (RE) |
| 数据 | HW_train_data.json (800条) |
| 评分 | Total = 0.4×NER_Score + 0.6×RE_Score |

## 运行说明

按顺序从上到下执行每个 Code Cell。

- **Cell 1**: 检查环境
- **Cell 2**: 安装依赖
- **Cell 3**: 查看数据分布
- **Cell 4**: 预下载模型 (~6-8 min)
- **Cell 5**: 训练 NER (~12 min, 8 epochs)
- **Cell 6**: 训练 RE (~25-35 min, 6 epochs)
- **Cell 7**: 联合推理 + 评估 (~3 min)
"""))

    # 标准 Cells
    cells.extend(make_cell_env_check())
    cells.extend(make_cell_install_deps())
    cells.extend(make_cell_data_check())
    cells.extend(make_cell_download_model())
    cells.extend(make_cell_train_ner(epochs=8, patience=2, batch_size=12, val_ratio=0.05))
    cells.extend(make_cell_train_re(epochs=6, neg_ratio=1, patience=2, batch_size=12, val_ratio=0.05))
    cells.extend(make_cell_predict(val_ratio=0.05))

    # 结尾
    cells.append(nbf.v4.new_markdown_cell("""### 完成！🎉

所有训练和推理已完成。关键输出文件:

| 文件 | 说明 |
|------|------|
| `output_ner/best_ner_model.pt` | NER 最佳模型 |
| `output_re/best_re_model.pt` | RE 最佳模型 |
| `predictions.json` | 预测结果 |

**查看最终分数**: 向上翻到 Cell 7 的输出末尾，找到 `Total Score` 行。
"""))

    nb.cells = cells
    return nb


def main():
    # 生成 Lite 版
    nb_lite = make_lite_notebook()
    nbf.write(nb_lite, "main_lite.ipynb")
    print("✅ Generated: main_lite.ipynb")

    # 生成完整版
    nb_full = make_full_notebook()
    nbf.write(nb_full, "main.ipynb")
    print("✅ Generated: main.ipynb")

    print("\nDone! 两个 Notebook 已生成。")
    print("如需修改，编辑 gen_notebooks.py 后重新运行即可。")


if __name__ == "__main__":
    main()
