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
TEST_FILE = "HW_test_data.json"


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


def make_cell_submit_test(re_threshold=0.9):
    """正式测试集一键推理 Cell"""
    md = f"""### Cell 8: 正式测试集 Pipeline 推理/评估，一键生成提交文件

使用已经训练好的 `output_ner` 和 `output_re`，对 `{TEST_FILE}` 生成提交文件。

默认和 Cell 7 一样：先让 NER 预测实体，再让 RE 预测关系。  
如果测试集里有 `entities/relations`，会自动加 `--evaluate` 打印分数。

如果你想直接使用测试集自带 `entities`，把 `USE_PROVIDED_TEST_ENTITIES = False` 改成 `True`。

**输出文件**: `predictions_submit.json`
"""
    code = f'''\
import json
from pathlib import Path

TEST_FILE = "{TEST_FILE}"
OUTPUT_FILE = "predictions_submit.json"
USE_PROVIDED_TEST_ENTITIES = False  # False = 和 Cell 7 一样走完整 NER→RE pipeline
EVALUATE_IF_LABELS_PRESENT = True
RE_THRESHOLD = {re_threshold}

assert Path(TEST_FILE).exists(), f"Missing {{TEST_FILE}}. 请把老师测试集放在当前目录。"
assert Path("./output_ner/best_ner_model.pt").exists(), "Missing ./output_ner/best_ner_model.pt"
assert Path("./output_re/best_re_model.pt").exists(), "Missing ./output_re/best_re_model.pt"

test_data = json.load(open(TEST_FILE, encoding="utf-8"))
assert isinstance(test_data, list) and len(test_data) > 0, "测试集应该是非空 list"
assert isinstance(test_data[0], dict) and "text" in test_data[0], "测试集每条样本应该有 text 字段"

provided_entities = sum(len(item.get("entities", [])) for item in test_data if isinstance(item, dict))
provided_relations = sum(len(item.get("relations", [])) for item in test_data if isinstance(item, dict))
print(f"Test samples: {{len(test_data)}}")
print(f"Provided entities: {{provided_entities}}")
print(f"Provided relations: {{provided_relations}}")

extra_flags = []
if USE_PROVIDED_TEST_ENTITIES and provided_entities > 0:
    extra_flags.append("--use_test_entities")
if EVALUATE_IF_LABELS_PRESENT and provided_entities > 0 and provided_relations > 0:
    extra_flags.append("--evaluate")

print("Use provided test entities:", USE_PROVIDED_TEST_ENTITIES and provided_entities > 0)
print("Evaluate if labels present:", "--evaluate" in extra_flags)
print("RE threshold:", RE_THRESHOLD)

cmd = (
    "python predict.py "
    "--ner_model_dir ./output_ner "
    "--re_model_dir ./output_re "
    f"--test_file {{TEST_FILE}} "
    f"--output {{OUTPUT_FILE}} "
    "--batch_size 12 "
    f"--re_threshold {{RE_THRESHOLD}} "
    "--max_length 384"
    + (" " + " ".join(extra_flags) if extra_flags else "")
)
print("Running:")
print(cmd)
exit_code = __import__("os").system(cmd)
assert exit_code == 0, f"推理失败，exit code={{exit_code}}"

preds = json.load(open(OUTPUT_FILE, encoding="utf-8"))
total_ent = sum(len(p.get("entities", [])) for p in preds)
total_rel = sum(len(p.get("relations", [])) for p in preds)

print("=" * 60)
print("Submit file ready")
print("=" * 60)
print(f"Output: {{Path(OUTPUT_FILE).resolve()}}")
print(f"Predictions: {{len(preds)}} texts, {{total_ent}} entities, {{total_rel}} relations")
print("请下载 predictions_submit.json 作为提交文件。")
'''
    return nbf.v4.new_markdown_cell(md), nbf.v4.new_code_cell(code)


def make_tools_notebook():
    """生成工具 Notebook：备份/还原/清理/打包。"""
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# FinalTest Tools

这个 Notebook 只做文件管理，不训练模型。

常用场景：
- 查看当前目录、模型、权重、预测文件是否存在
- 备份 `./deberta-v3-large`
- 从备份还原 `./deberta-v3-large`
- 删除已有 NER/RE 权重，重新训练前清场
- 打包 `predictions.json` 和输出模型，方便下载

安全规则：删除/还原类 Cell 默认不会执行，必须把对应的 `CONFIRM_...` 改成 `True`。
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 1: 状态检查"))
    cells.append(nbf.v4.new_code_cell("""\
from pathlib import Path
import json
import os
import shutil
import time
import zipfile

ROOT = Path(".").resolve()
BASE_MODEL = ROOT / "deberta-v3-large"
BACKUP_ROOT = ROOT / "model_backups"
NER_OUT = ROOT / "output_ner"
RE_OUT = ROOT / "output_re"
PRED_FILE = ROOT / "predictions.json"

def format_size(num_bytes):
    if num_bytes is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

def path_size(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def show_path(path):
    path = Path(path)
    status = "DIR" if path.is_dir() else "FILE" if path.is_file() else "MISSING"
    print(f"{status:7s} {str(path):35s} {format_size(path_size(path))}")

print("ROOT:", ROOT)
print()
for p in [BASE_MODEL, NER_OUT, RE_OUT, PRED_FILE, BACKUP_ROOT]:
    show_path(p)

print("\\nPossible base-model backups:")
candidates = []
if BACKUP_ROOT.exists():
    candidates.extend(sorted(BACKUP_ROOT.iterdir()))
candidates.extend(sorted(ROOT.glob("deberta-v3-large_backup*")))
if candidates:
    for p in candidates:
        show_path(p)
else:
    print("  No backups found yet.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 2: 备份基座模型"))
    cells.append(nbf.v4.new_code_cell("""\
# 把 ./deberta-v3-large 复制到 ./model_backups/ 下。
# 这是非破坏性操作，但会占用几 GB 磁盘空间。

CONFIRM_BACKUP_BASE_MODEL = False
BACKUP_NAME = ""  # 留空则自动用时间戳；也可以写 "deberta-v3-large_backup_a10_ok"

if not CONFIRM_BACKUP_BASE_MODEL:
    print("Not running. Set CONFIRM_BACKUP_BASE_MODEL = True to backup the base model.")
else:
    assert BASE_MODEL.exists() and BASE_MODEL.is_dir(), f"Base model folder not found: {BASE_MODEL}"
    BACKUP_ROOT.mkdir(exist_ok=True)
    name = BACKUP_NAME.strip() or f"deberta-v3-large_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    target = BACKUP_ROOT / name
    assert not target.exists(), f"Backup already exists: {target}"
    print(f"Copying {BASE_MODEL} -> {target}")
    shutil.copytree(BASE_MODEL, target)
    show_path(target)
    print("Backup complete.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 3: 从备份还原基座模型"))
    cells.append(nbf.v4.new_code_cell("""\
# 从某个备份还原 ./deberta-v3-large。
# 为安全起见，当前 ./deberta-v3-large 不会直接丢弃，会先挪到 *_before_restore_时间戳。

CONFIRM_RESTORE_BASE_MODEL = False
RESTORE_FROM = ""  # 例: "model_backups/deberta-v3-large_backup_20260521_120000"

if not CONFIRM_RESTORE_BASE_MODEL:
    print("Not running. Set CONFIRM_RESTORE_BASE_MODEL = True and RESTORE_FROM to restore.")
else:
    source = Path(RESTORE_FROM).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    assert source.exists() and source.is_dir(), f"Backup folder not found: {source}"

    if BASE_MODEL.exists():
        parked = ROOT / f"deberta-v3-large_before_restore_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"Moving current base model to: {parked}")
        shutil.move(str(BASE_MODEL), str(parked))

    print(f"Restoring {source} -> {BASE_MODEL}")
    shutil.copytree(source, BASE_MODEL)
    show_path(BASE_MODEL)
    print("Restore complete.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 4: 删除已有训练权重和预测文件"))
    cells.append(nbf.v4.new_code_cell("""\
# 清理训练输出。适合正式重跑前使用。
# 注意：删除 output_ner/output_re 会删除已经训练好的权重。

CONFIRM_DELETE_OUTPUTS = False
DELETE_NER = True
DELETE_RE = True
DELETE_PREDICTIONS = True
DELETE_BATCH_TEST_DIRS = True

targets = []
if DELETE_NER:
    targets.append(NER_OUT)
if DELETE_RE:
    targets.append(RE_OUT)
if DELETE_PREDICTIONS:
    targets.append(PRED_FILE)
if DELETE_BATCH_TEST_DIRS:
    targets.extend([ROOT / "tmp_batchtest_ner", ROOT / "tmp_batchtest_re"])

print("Targets:")
for t in targets:
    show_path(t)

if not CONFIRM_DELETE_OUTPUTS:
    print("\\nNot deleting. Set CONFIRM_DELETE_OUTPUTS = True to delete these targets.")
else:
    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
            print("Deleted dir:", t)
        elif t.is_file():
            t.unlink()
            print("Deleted file:", t)
        else:
            print("Skip missing:", t)
    print("Cleanup complete.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 5: 打包结果，方便下载"))
    cells.append(nbf.v4.new_code_cell("""\
# 把 predictions.json、output_ner、output_re 打成 zip。
# 如果只想提交 predictions.json，也可以只下载那个文件。

ARCHIVE_NAME = f"finaltest_outputs_{time.strftime('%Y%m%d_%H%M%S')}.zip"
archive_path = ROOT / ARCHIVE_NAME

include_paths = [PRED_FILE, NER_OUT, RE_OUT]
existing = [p for p in include_paths if p.exists()]
assert existing, "No output files found to archive."

with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in existing:
        if p.is_file():
            zf.write(p, p.relative_to(ROOT))
        else:
            for child in p.rglob("*"):
                if child.is_file():
                    zf.write(child, child.relative_to(ROOT))

show_path(archive_path)
print("Archive created.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 6: 清理 HuggingFace 坏缓存"))
    cells.append(nbf.v4.new_code_cell("""\
# 下载中断后，HF cache 偶尔会留下坏缓存。一般不需要跑。
# 这个 Cell 只删缓存，不删 ./deberta-v3-large。

CONFIRM_CLEAR_HF_CACHE = False
cache_targets = [
    Path("/root/.cache/huggingface/hub/models--microsoft--deberta-v3-large"),
    Path.home() / ".cache/huggingface/hub/models--microsoft--deberta-v3-large",
]

for p in cache_targets:
    show_path(p)

if not CONFIRM_CLEAR_HF_CACHE:
    print("\\nNot clearing cache. Set CONFIRM_CLEAR_HF_CACHE = True to clear.")
else:
    for p in cache_targets:
        if p.exists():
            shutil.rmtree(p)
            print("Cleared:", p)
    print("Cache cleanup complete.")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 7: 当前推荐训练命令"))
    cells.append(nbf.v4.new_code_cell("""\
print("NER full training:")
print("python train_ner.py --data HW_train_data.json --model ./deberta-v3-large --save_dir ./output_ner --max_length 384 --batch_size 12 --epochs 8 --lr 1e-5 --warmup_ratio 0.1 --dropout 0.1 --val_ratio 0.05 --patience 2")
print()
print("RE full training:")
print("python train_re.py --data HW_train_data.json --model ./deberta-v3-large --save_dir ./output_re --max_length 384 --batch_size 12 --epochs 6 --lr 1e-5 --warmup_ratio 0.1 --dropout 0.1 --neg_ratio 1 --val_ratio 0.05 --patience 2")
print()
print("Continue RE from current best to epoch 8:")
print("python train_re.py --data HW_train_data.json --model ./deberta-v3-large --save_dir ./output_re --max_length 384 --batch_size 12 --epochs 8 --lr 1e-5 --warmup_ratio 0.1 --dropout 0.1 --neg_ratio 1 --val_ratio 0.05 --patience 2 --resume_from ./output_re/best_re_model.pt")
print()
print("Inference:")
print("python predict.py --ner_model_dir ./output_ner --re_model_dir ./output_re --data HW_train_data.json --output predictions.json --batch_size 12 --re_threshold 0.9 --max_length 384 --val_ratio 0.05 --evaluate")
"""))

    cells.append(nbf.v4.new_markdown_cell("### Cell 8: 继续训练 RE 到第 8 轮"))
    cells.append(nbf.v4.new_code_cell("""\
# 如果 RE best checkpoint 是第 6 轮，这个 Cell 会继续跑第 7-8 轮。
# 如果第 7/8 轮没有更好，旧的 best_re_model.pt 会保留。

CONFIRM_CONTINUE_RE = False
TARGET_EPOCHS = 8

if not CONFIRM_CONTINUE_RE:
    print("Not running. Set CONFIRM_CONTINUE_RE = True to continue RE training.")
else:
    import subprocess
    from pathlib import Path

    ROOT = Path(".").resolve()
    RE_OUT = ROOT / "output_re"

    assert (RE_OUT / "best_re_model.pt").exists(), "Missing ./output_re/best_re_model.pt"
    cmd = [
        "python", "train_re.py",
        "--data", "HW_train_data.json",
        "--model", "./deberta-v3-large",
        "--save_dir", "./output_re",
        "--max_length", "384",
        "--batch_size", "12",
        "--epochs", str(TARGET_EPOCHS),
        "--lr", "1e-5",
        "--warmup_ratio", "0.1",
        "--dropout", "0.1",
        "--neg_ratio", "1",
        "--val_ratio", "0.05",
        "--patience", "2",
        "--resume_from", "./output_re/best_re_model.pt",
    ]
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Continue RE training complete.")
"""))

    nb.cells = cells
    return nb


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
- **Cell 8**: 正式测试集推理，生成 `predictions_submit.json`
"""))

    # 标准 Cells
    cells.extend(make_cell_env_check())
    cells.extend(make_cell_install_deps())
    cells.extend(make_cell_data_check())
    cells.extend(make_cell_download_model())
    cells.extend(make_cell_train_ner(epochs=8, patience=2, batch_size=12, val_ratio=0.05))
    cells.extend(make_cell_train_re(epochs=6, neg_ratio=1, patience=2, batch_size=12, val_ratio=0.05))
    cells.extend(make_cell_predict(val_ratio=0.05))
    cells.extend(make_cell_submit_test(re_threshold=0.9))

    # 结尾
    cells.append(nbf.v4.new_markdown_cell("""### 完成！🎉

所有训练和推理已完成。关键输出文件:

| 文件 | 说明 |
|------|------|
| `output_ner/best_ner_model.pt` | NER 最佳模型 |
| `output_re/best_re_model.pt` | RE 最佳模型 |
| `predictions.json` | 预测结果 |
| `predictions_submit.json` | 正式测试集提交文件 |

**查看验证分数**: 向上翻到 Cell 7 的输出末尾，找到 `Total Score` 行。  
**正式提交**: 下载 Cell 8 生成的 `predictions_submit.json`。
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

    # 生成工具 Notebook
    nb_tools = make_tools_notebook()
    nbf.write(nb_tools, "tools.ipynb")
    print("✅ Generated: tools.ipynb")

    print("\nDone! Notebooks 已生成。")
    print("如需修改，编辑 gen_notebooks.py 后重新运行即可。")


if __name__ == "__main__":
    main()
