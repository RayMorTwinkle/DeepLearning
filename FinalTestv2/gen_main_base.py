"""
Generate a single-file base-model notebook for FinalTestv2.

The generated notebook embeds the runtime Python modules. It trains only from
HW_train_data.json and can use labeled HW_test_data.json as a fixed validation
and output set.
"""
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "FinalTestCloud"
OUT_DIR = ROOT / "FinalTestv2"

TRAIN_DATA_FILE = "HW_train_data.json"
TEST_DATA_FILE = "HW_test_data.json"
MODEL_NAME = "microsoft/deberta-v3-base"
LOCAL_MODEL_DIR = "./deberta-v3-base"

SCRIPT_FILES = [
    "data_utils.py",
    "ner_model.py",
    "re_model.py",
    "train_ner.py",
    "train_re.py",
    "predict.py",
]


def read_scripts():
    scripts = {}
    for name in SCRIPT_FILES:
        scripts[name] = (CLOUD / name).read_text(encoding="utf-8")
    return scripts


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def make_notebook():
    scripts = read_scripts()
    scripts_literal = json.dumps(scripts, ensure_ascii=False, indent=2)

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(f"""# FinalTest v2 - Base 小模型单文件实验线

这是一套备用实验，不替换你已经跑通的 large 主线。

| 目标 | 方案 |
|------|------|
| 小一点的模型 | `{MODEL_NAME}` |
| NER 架构 | DeBERTa-v3-base + CRF |
| RE 架构 | DeBERTa-v3-base + Entity Markers |
| 参数策略 | 更大 batch、更高 lr、更多 epoch、early stopping |
| 输出 | `predictions_base.json` |

为什么这算“强而小”的路线：
- DeBERTa-v3-base 是 strong encoder，比 large 小很多，训练/推理更快。
- NER 仍然用 CRF，适合 BIOES 这种连续标签。
- RE 仍然用实体标记，让模型明确知道“哪两个实体要判断关系”。
- 最后扫 `re_threshold`，因为你的关系误报是当前 pipeline 的最大瓶颈。

云端最少上传：
- `main_base.ipynb`
- `HW_train_data.json`

如果 `HW_test_data.json` 是带标注的验证集，也一起上传。Notebook 会把它作为验证/输出数据，不会把它加入训练。

按顺序运行 Cell。不要一次性 Run All。
"""))

    cells.append(md("### Cell 1: 环境与依赖"))
    cells.append(code(f"""\
import os, sys, json, shutil, subprocess, textwrap
from pathlib import Path

# Colab usually reaches huggingface.co directly. Do not set hf-mirror here.
os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"

print("=" * 60)
print("Base Notebook Environment")
print("=" * 60)
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU Memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
except Exception as e:
    print("Torch import failed before install:", repr(e))

print("Files:", sorted(os.listdir(".")))
print("=" * 60)

!pip install "transformers<5" sentencepiece protobuf pytorch-crf scikit-learn tqdm huggingface_hub -q

import torch, transformers
print(f"torch {{torch.__version__}}, transformers {{transformers.__version__}}")
print("CUDA available:", torch.cuda.is_available())
print("Dependencies OK")
"""))

    cells.append(md("### Cell 2: 数据检查"))
    cells.append(code(f"""\
import json
from collections import Counter
from pathlib import Path

def find_data_file(filename):
    candidates = [
        filename,
        f"sample_data/{{filename}}",
        f"/content/{{filename}}",
        f"/content/sample_data/{{filename}}",
    ]
    return next((p for p in candidates if Path(p).exists()), None)

def require_labeled_dataset(data, name):
    assert isinstance(data, list) and data, f"{{name}} must be a non-empty JSON list."
    for i, item in enumerate(data[:5]):
        assert isinstance(item, dict) and "text" in item, f"{{name}} item {{i}} missing text."
        assert "entities" in item and "relations" in item, (
            f"{{name}} item {{i}} is not labeled. It cannot be used as validation data."
        )

TRAIN_DATA_FILE = find_data_file("{TRAIN_DATA_FILE}")
VALIDATION_DATA_FILE = find_data_file("{TEST_DATA_FILE}")
assert TRAIN_DATA_FILE is not None, (
    "Missing HW_train_data.json. Upload it next to this notebook or into sample_data/."
)

train_data = json.load(open(TRAIN_DATA_FILE, encoding="utf-8"))
require_labeled_dataset(train_data, TRAIN_DATA_FILE)

print("Primary train file:", TRAIN_DATA_FILE, "samples:", len(train_data))

if VALIDATION_DATA_FILE is not None:
    validation_data = json.load(open(VALIDATION_DATA_FILE, encoding="utf-8"))
    require_labeled_dataset(validation_data, VALIDATION_DATA_FILE)
    print("Validation file:", VALIDATION_DATA_FILE, "samples:", len(validation_data))
else:
    validation_data = None
    print("No labeled HW_test_data.json found. Training scripts will split validation from training data.")

DATA_FILE = TRAIN_DATA_FILE
EVAL_DATA_FILE = VALIDATION_DATA_FILE or DATA_FILE
print("Training data file:", DATA_FILE)
print("Evaluation data file:", EVAL_DATA_FILE)

def show_counts(title, data):
    ent_counter = Counter()
    rel_counter = Counter()
    for item in data:
        for e in item.get("entities", []):
            ent_counter[e["label"]] += 1
        for r in item.get("relations", []):
            rel_counter[r["label"]] += 1
    print(title)
    print("  样本数:", len(data))
    print("  实体:", sum(ent_counter.values()), dict(ent_counter.most_common()))
    print("  关系:", sum(rel_counter.values()), dict(rel_counter.most_common()))

show_counts("Train data:", train_data)
if validation_data is not None:
    show_counts("Validation data:", validation_data)
print("Data OK")
"""))

    cells.append(md("### Cell 3: 写出运行脚本\n\n这个 Cell 会从 Notebook 内部生成 `.py` 文件，云端不需要另外上传脚本。"))
    cells.append(code(f"""\
from pathlib import Path

SCRIPTS = {scripts_literal}

for name, content in SCRIPTS.items():
    Path(name).write_text(content, encoding="utf-8")
    print(f"Wrote {{name}} ({{len(content):,}} chars)")

print("Runtime scripts ready.")
"""))

    cells.append(md("### Cell 4: 下载 base 基座模型\n\nCPU 环境可以先跑这个 Cell；训练前再切 GPU。"))
    cells.append(code(f"""\
import os, shutil
from pathlib import Path

# Colab: direct HuggingFace is normally more stable than hf-mirror.
os.environ.pop("HF_ENDPOINT", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

MODEL_NAME = "{MODEL_NAME}"
LOCAL_MODEL_DIR = "{LOCAL_MODEL_DIR}"

model_dir = Path(LOCAL_MODEL_DIR)
required = [model_dir / "config.json", model_dir / "tokenizer_config.json"]
has_weights = any((model_dir / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
complete = model_dir.exists() and all(p.exists() for p in required) and has_weights

if complete:
    print(f"{{LOCAL_MODEL_DIR}} already looks complete, skip download.")
else:
    shutil.rmtree(LOCAL_MODEL_DIR, ignore_errors=True)
    shutil.rmtree("/root/.cache/huggingface/hub/models--microsoft--deberta-v3-base", ignore_errors=True)
    print(f"Downloading {{MODEL_NAME}} -> {{LOCAL_MODEL_DIR}}")
    snapshot_download(
        MODEL_NAME,
        local_dir=LOCAL_MODEL_DIR,
        force_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )
    print("Downloaded.")
"""))

    cells.append(md("""### Cell 5: 训练 NER base

参数选择：

| 参数 | 值 | 原因 |
|------|----|------|
| batch_size | 24 | base 比 large 小，A10 通常能承受 |
| epochs | 10 | 给小模型多一点学习机会 |
| lr | 2e-5 | base 常用微调学习率，比 large 的 1e-5 激进一点 |
| dropout | 0.15 | 小数据防过拟合 |
| patience | 3 | 连续 3 轮不涨就停 |
"""))
    cells.append(code(f"""\
import torch, os, subprocess
assert torch.cuda.is_available(), "CUDA=False：训练前请切换到 GPU 环境。"

cmd = [
    "python", "train_ner.py",
    "--data", DATA_FILE,
    "--model", "{LOCAL_MODEL_DIR}",
    "--save_dir", "./output_ner_base",
    "--max_length", "384",
    "--batch_size", "24",
    "--epochs", "10",
    "--lr", "2e-5",
    "--warmup_ratio", "0.1",
    "--dropout", "0.15",
    "--val_ratio", "0.05",
    "--patience", "3",
]
if VALIDATION_DATA_FILE:
    cmd.extend(["--val_data", VALIDATION_DATA_FILE])

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

assert os.path.exists("./output_ner_base/best_ner_model.pt"), "NER base model not saved."
print(f"NER base model: {{os.path.getsize('./output_ner_base/best_ner_model.pt')/1e9:.2f}} GB")
"""))

    cells.append(md("""### Cell 6: 训练 RE base

关系抽取是当前短板，所以 base 线给 RE 多跑几轮。

| 参数 | 值 | 原因 |
|------|----|------|
| batch_size | 24 | base 更省显存 |
| epochs | 10 | RE 第 6 轮仍可能上涨，base 给到 10 |
| neg_ratio | 1 | 正负 1:1，减少 NONE 压倒正类 |
| lr | 2e-5 | base 微调常用 |
| dropout | 0.15 | 防止记住训练集 |
"""))
    cells.append(code(f"""\
import torch, os, subprocess
assert torch.cuda.is_available(), "CUDA=False：训练前请切换到 GPU 环境。"

cmd = [
    "python", "train_re.py",
    "--data", DATA_FILE,
    "--model", "{LOCAL_MODEL_DIR}",
    "--save_dir", "./output_re_base",
    "--max_length", "384",
    "--batch_size", "24",
    "--epochs", "10",
    "--lr", "2e-5",
    "--warmup_ratio", "0.1",
    "--dropout", "0.15",
    "--neg_ratio", "1",
    "--val_ratio", "0.05",
    "--patience", "3",
]
if VALIDATION_DATA_FILE:
    cmd.extend(["--val_data", VALIDATION_DATA_FILE])

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

assert os.path.exists("./output_re_base/best_re_model.pt"), "RE base model not saved."
print(f"RE base model: {{os.path.getsize('./output_re_base/best_re_model.pt')/1e9:.2f}} GB")
"""))

    cells.append(md("""### Cell 7: 扫 RE 阈值，自动选验证集最高分

这一步会在 `HW_test_data.json` 验证集上试几个阈值。  
如果 `0.9` 还是最好，就说明 base 也有类似的“关系误报偏多”问题。
"""))
    cells.append(code(f"""\
import subprocess, re, shutil
from pathlib import Path

thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
results = []

for thr in thresholds:
    out_file = f"predictions_base_thr_{{thr:.2f}}.json"
    cmd = [
        "python", "predict.py",
        "--ner_model_dir", "./output_ner_base",
        "--re_model_dir", "./output_re_base",
        "--test_file", EVAL_DATA_FILE,
        "--output", out_file,
        "--batch_size", "24",
        "--re_threshold", str(thr),
        "--max_length", "384",
        "--evaluate",
    ]
    print("=" * 80)
    print("Testing threshold", thr)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"predict.py failed at threshold {{thr}}")

    m = re.search(r"Total Score = .* = ([0-9.]+)", proc.stdout)
    total = float(m.group(1)) if m else -1.0
    results.append((total, thr, out_file))

results.sort(reverse=True)
best_total, best_thr, best_file = results[0]
shutil.copy2(best_file, "predictions_base.json")

print("\\nThreshold sweep summary:")
for total, thr, out_file in results:
    print(f"thr={{thr:.2f}} total={{total:.4f}} file={{out_file}}")

print(f"\\nBest threshold: {{best_thr:.2f}}, Total Score: {{best_total:.4f}}")
print("Saved best validation prediction to predictions_base.json")
"""))

    cells.append(md("""### Cell 8: 对 `HW_test_data.json` 生成输出

这个 Cell 默认读取 `HW_test_data.json`，输出 `predictions_test.json`。

如果 `HW_test_data.json` 里已经有 `entities`，默认会直接使用这些实体，只预测关系。  
如果你想模拟真正测试集流程，把 `USE_PROVIDED_ENTITIES = False`，它会重新跑 NER + RE。
"""))
    cells.append(code("""\
from pathlib import Path
import subprocess
import json

def find_data_file(filename):
    candidates = [
        filename,
        f"sample_data/{filename}",
        f"/content/{filename}",
        f"/content/sample_data/{filename}",
    ]
    return next((p for p in candidates if Path(p).exists()), None)

TEST_FILE = find_data_file("HW_test_data.json")
OUTPUT_FILE = "predictions_test.json"
BEST_THRESHOLD = 0.85  # 可以改成 Cell 7 打印出来的最佳 threshold
USE_PROVIDED_ENTITIES = True

assert TEST_FILE is not None, "Missing HW_test_data.json."

preview = json.load(open(TEST_FILE, encoding="utf-8"))
has_entities = (
    isinstance(preview, list)
    and preview
    and isinstance(preview[0], dict)
    and "entities" in preview[0]
)

cmd = [
    "python", "predict.py",
    "--ner_model_dir", "./output_ner_base",
    "--re_model_dir", "./output_re_base",
    "--test_file", TEST_FILE,
    "--output", OUTPUT_FILE,
    "--batch_size", "24",
    "--re_threshold", str(BEST_THRESHOLD),
    "--max_length", "384",
]
if USE_PROVIDED_ENTITIES and has_entities:
    cmd.append("--use_test_entities")

print("Using test file:", TEST_FILE)
print("Output file:", OUTPUT_FILE)
print("Use provided entities:", USE_PROVIDED_ENTITIES and has_entities)
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("Saved", OUTPUT_FILE)
"""))

    cells.append(md("### Cell 9: 打包 base 实验结果"))
    cells.append(code("""\
from pathlib import Path
import zipfile, time

archive = Path(f"base_outputs_{time.strftime('%Y%m%d_%H%M%S')}.zip")
include = [Path("predictions_base.json"), Path("output_ner_base"), Path("output_re_base")]
existing = [p for p in include if p.exists()]
assert existing, "No base outputs found."

with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in existing:
        if p.is_file():
            zf.write(p, p.name)
        else:
            for child in p.rglob("*"):
                if child.is_file():
                    zf.write(child, child)

print("Created:", archive, archive.stat().st_size / 1e9, "GB")
"""))

    nb.cells = cells
    return nb


def main():
    OUT_DIR.mkdir(exist_ok=True)
    nb = make_notebook()
    out = OUT_DIR / "main_base.ipynb"
    nbf.write(nb, out)
    print(f"Generated {out}")


if __name__ == "__main__":
    main()
