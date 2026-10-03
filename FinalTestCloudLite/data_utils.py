"""
FinalTest - 数据处理工具模块
    命名实体识别（NER）+ 关系抽取（RE）
"""
import json
import random
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

# ============================================================
# 常量定义
# ============================================================

ENTITY_TYPES = [
    "CROP", "VAR", "TRT", "GST", "GENE", "QTL",
    "MRK", "CHR", "BM", "CROSS", "ABS", "BIS"
]

RELATION_TYPES = ["CON", "USE", "HAS", "AFF", "OCI", "LOI"]

# BIOES 标签前缀
BIOES_PREFIXES = ["B", "I", "E", "S"]

# 构建 label -> id 映射
def build_ner_label_map(entity_types: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    构建 BIOES 标签映射
    Returns: (label2id, id2label)
    """
    label2id = {"O": 0}
    idx = 1
    for etype in entity_types:
        for prefix in BIOES_PREFIXES:
            label2id[f"{prefix}-{etype}"] = idx
            idx += 1
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

NER_LABEL2ID, NER_ID2LABEL = build_ner_label_map(ENTITY_TYPES)
NUM_NER_LABELS = len(NER_LABEL2ID)

# 关系标签映射
REL_LABEL2ID = {"NONE": 0}
for i, rtype in enumerate(RELATION_TYPES, start=1):
    REL_LABEL2ID[rtype] = i
REL_ID2LABEL = {v: k for k, v in REL_LABEL2ID.items()}
NUM_REL_LABELS = len(REL_LABEL2ID)


# ============================================================
# 数据加载
# ============================================================

def load_data(filepath: str) -> List[Dict]:
    """加载 JSON 训练数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_data(data: List[Dict], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """划分训练集和验证集"""
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = [data[i] for i in indices[:split_idx]]
    val_data = [data[i] for i in indices[split_idx:]]
    return train_data, val_data


# ============================================================
# NER 数据处理
# ============================================================

def entities_to_bioes(text: str, entities: List[Dict]) -> List[str]:
    """
    将实体列表转换为 BIOES 字符级标签序列
    """
    chars = list(text)
    labels = ["O"] * len(chars)

    for ent in entities:
        start, end = ent["start"], ent["end"]
        ent_type = ent["label"]
        if end - start == 1:
            labels[start] = f"S-{ent_type}"
        else:
            labels[start] = f"B-{ent_type}"
            for i in range(start + 1, end - 1):
                labels[i] = f"I-{ent_type}"
            labels[end - 1] = f"E-{ent_type}"

    return labels


def tokenize_and_align_ner(
    tokenizer: PreTrainedTokenizer,
    text: str,
    char_labels: List[str],
    max_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对文本进行 tokenize 并将字符级 BIOES 标签对齐到 token 级

    Returns:
        input_ids: [seq_len]
        attention_mask: [seq_len]
        token_labels: [seq_len] 带 -100 标记忽略位置
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    offset_mapping = encoding["offset_mapping"][0]

    token_labels = []
    for token_idx in range(len(input_ids)):
        char_start, char_end = offset_mapping[token_idx].tolist()

        # 特殊 token（[CLS], [SEP], [PAD]）：忽略
        if char_start == char_end == 0:
            token_labels.append(-100)
            continue

        # 取该 token 覆盖的字符范围的标签
        token_char_labels = char_labels[char_start:char_end]
        # 过滤掉 O 标签，看是否有实体标签
        entity_labels = [l for l in token_char_labels if l != "O"]

        if not entity_labels:
            token_labels.append(NER_LABEL2ID["O"])
        else:
            # 取出现最多的实体标签
            # 对于子词 token，可能同时包含 B- 和 I-，优先保留 B- 或 S-
            has_s = any(l.startswith("S-") for l in entity_labels)
            has_b = any(l.startswith("B-") for l in entity_labels)
            has_e = any(l.startswith("E-") for l in entity_labels)
            has_i = any(l.startswith("I-") for l in entity_labels)

            if has_s:
                chosen = [l for l in entity_labels if l.startswith("S-")][0]
            elif has_b:
                chosen = [l for l in entity_labels if l.startswith("B-")][0]
            elif has_e:
                chosen = [l for l in entity_labels if l.startswith("E-")][0]
            else:
                chosen = entity_labels[0]

            token_labels.append(NER_LABEL2ID[chosen])

    # padding 对齐长度
    token_labels = token_labels[:max_length]
    while len(token_labels) < max_length:
        token_labels.append(-100)

    return input_ids, attention_mask, torch.tensor(token_labels, dtype=torch.long), offset_mapping


class NERDataset(Dataset):
    """NER 数据集"""

    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for item in data:
            text = item["text"]
            entities = item["entities"]
            char_labels = entities_to_bioes(text, entities)
            input_ids, attention_mask, token_labels, offset_mapping = tokenize_and_align_ner(
                tokenizer, text, char_labels, max_length
            )
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": token_labels,
                "offset_mapping": offset_mapping,
                "text": text,
                "gold_entities": entities,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def tokens_to_entities_with_offsets(
    input_ids: torch.Tensor,
    label_ids: list,
    offset_mapping: torch.Tensor,
    original_text: str,
) -> List[Dict]:
    """
    将 token 级 BIOES 标签结合 offset_mapping 转换回字符级实体

    Args:
        input_ids: token IDs
        label_ids: 标签 ID 列表
        offset_mapping: 每个 token 的 [char_start, char_end]
        original_text: 原始文本

    Returns:
        entities: [{"start": int, "end": int, "text": str, "label": str}, ...]
    """
    entities = []
    current_entity = None

    def normalize_span(start: int, end: int) -> Tuple[int, int]:
        """Fast tokenizers for SentencePiece may include surrounding spaces in offsets."""
        while start < end and original_text[start].isspace():
            start += 1
        while end > start and original_text[end - 1].isspace():
            end -= 1
        return start, end

    def append_current():
        if current_entity is None:
            return
        start, end = normalize_span(current_entity["start"], current_entity["end"])
        if start >= end:
            return
        ent_text = original_text[start:end]
        entities.append({
            "start": start,
            "end": end,
            "text": ent_text,
            "label": current_entity["type"],
        })

    for token_idx, (token_id, label_id, (char_start, char_end)) in enumerate(
        zip(input_ids, label_ids, offset_mapping)
    ):
        token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
        char_start = int(char_start) if hasattr(char_start, 'item') else char_start
        char_end = int(char_end) if hasattr(char_end, 'item') else char_end

        if label_id == -100:
            continue

        # 跳过特殊 token（[CLS], [SEP], [PAD] 的 offset 为 (0,0)）
        if char_start == char_end == 0:
            continue

        label = NER_ID2LABEL.get(label_id, "O")

        if label == "O":
            if current_entity is not None:
                append_current()
                current_entity = None
            continue

        prefix, etype = label.split("-", 1)

        if prefix == "S":
            if current_entity is not None:
                append_current()
                current_entity = None
            start, end = normalize_span(char_start, char_end)
            if start < end:
                entities.append({
                    "start": start,
                    "end": end,
                    "text": original_text[start:end],
                    "label": etype,
                })
            continue

        if prefix == "B":
            if current_entity is not None:
                append_current()
            current_entity = {"start": char_start, "end": char_end, "type": etype}

        elif prefix in ("I", "E"):
            if current_entity is not None and current_entity["type"] == etype:
                current_entity["end"] = char_end
                if prefix == "E":
                    append_current()
                    current_entity = None
            else:
                if current_entity is not None:
                    append_current()
                current_entity = {"start": char_start, "end": char_end, "type": etype}
                if prefix == "E":
                    append_current()
                    current_entity = None

    # 末尾未关闭的实体
    if current_entity is not None:
        append_current()

    return entities


# ============================================================
# RE 数据处理（Entity Markers 方案）
# ============================================================

def insert_entity_markers(
    text: str,
    head: Dict,
    tail: Dict,
) -> str:
    """
    在文本中插入实体标记 [E1]...[/E1] 和 [E2]...[/E2]
    从右向左插入，确保偏移正确
    """
    # head 和 tail 均有 start, end 字段
    h_start, h_end = head["start"], head["end"]
    t_start, t_end = tail["start"], tail["end"]

    # 按右向左顺序排列插入点
    insertions = [
        (h_start, "[E1] "),
        (h_end, " [/E1]"),
        (t_start, "[E2] "),
        (t_end, " [/E2]"),
    ]
    # 从右到左排序
    insertions.sort(key=lambda x: x[0], reverse=True)

    result = text
    for pos, marker in insertions:
        result = result[:pos] + marker + result[pos:]

    return result


def prepare_re_samples(
    data: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 384,
    neg_ratio: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    为 RE 任务准备训练样本

    每条文本生成多个（头实体, 尾实体, 关系标签）样本
    正样本：来自标注的关系
    负样本：从无关系的实体对中随机采样（比例 neg_ratio:1）

    Returns:
        samples: list of {"input_ids", "attention_mask", "label_id"}
    """
    import random as _random
    _random.seed(seed)

    samples = []

    # 关系 token 类型映射
    rel_type_map = {
        "CONTAINS": "CON",
        "USES": "USE",
        "HAS": "HAS",
        "AFFECTS": "AFF",
        "OCCURS_IN": "OCI",
        "LOCATED_IN": "LOI",
    }

    for item in data:
        text = item["text"]
        entities = item["entities"]
        relations = item["relations"]

        if len(entities) < 2:
            continue

        # 构建正样本关系集合：(head_start, head_end, tail_start, tail_end) -> label
        positive_pairs = {}
        for rel in relations:
            # 统一关系标签
            raw_label = rel["label"]
            label = rel_type_map.get(raw_label, raw_label)
            key = (rel["head_start"], rel["head_end"], rel["tail_start"], rel["tail_end"])
            positive_pairs[key] = label

        # 生成所有实体对
        all_pairs = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                pair_key = (e1["start"], e1["end"], e2["start"], e2["end"])
                is_positive = pair_key in positive_pairs
                all_pairs.append((e1, e2, is_positive, pair_key))

        # 分离正负样本
        pos_pairs = [(e1, e2, k) for e1, e2, is_pos, k in all_pairs if is_pos]
        neg_pairs = [(e1, e2, k) for e1, e2, is_pos, k in all_pairs if not is_pos]

        # 正样本全部保留
        for e1, e2, pair_key in pos_pairs:
            label = positive_pairs[pair_key]
            marked_text = insert_entity_markers(text, e1, e2)
            encoding = tokenizer(
                marked_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            samples.append({
                "input_ids": encoding["input_ids"][0],
                "attention_mask": encoding["attention_mask"][0],
                "label_id": REL_LABEL2ID[label],
            })

        # 负样本：随机采样 neg_ratio * n_pos
        n_neg_sample = min(len(neg_pairs), len(pos_pairs) * neg_ratio)
        if n_neg_sample > 0:
            sampled_neg = _random.sample(neg_pairs, n_neg_sample)
            for e1, e2, _ in sampled_neg:
                marked_text = insert_entity_markers(text, e1, e2)
                encoding = tokenizer(
                    marked_text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                samples.append({
                    "input_ids": encoding["input_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                    "label_id": REL_LABEL2ID["NONE"],
                })

    # 打乱
    _random.shuffle(samples)
    return samples


class REDataset(Dataset):
    """RE 数据集"""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Tokenizer 工具
# ============================================================

def add_entity_marker_tokens(tokenizer: PreTrainedTokenizer):
    """为 tokenizer 添加实体标记特殊 token: [E1], [/E1], [E2], [/E2]"""
    special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


# ============================================================
# 评估指标
# ============================================================

def compute_ner_metrics(
    pred_entities: List[List[Dict]],
    gold_entities: List[List[Dict]],
) -> Dict[str, float]:
    """
    计算 NER 的 Precision, Recall, F1

    Args:
        pred_entities: 每条文本预测的实体列表 [{start, end, text, label}, ...]
        gold_entities: 每条文本标注的实体列表
    """
    tp = fp = fn = 0

    for preds, golds in zip(pred_entities, gold_entities):
        pred_set = set((e["start"], e["end"], e["label"]) for e in preds)
        gold_set = set((e["start"], e["end"], e["label"]) for e in golds)

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_re_metrics(
    pred_relations: List[List[Dict]],
    gold_relations: List[List[Dict]],
) -> Dict[str, float]:
    """
    计算 RE 的 Precision, Recall, F1

    Args:
        pred_relations: 每条文本预测的关系列表
            [{head_start, head_end, head_type, tail_start, tail_end, tail_type, label}, ...]
        gold_relations: 每条文本标注的关系列表
    """
    tp = fp = fn = 0

    for preds, golds in zip(pred_relations, gold_relations):
        pred_set = set(
            (r["head_start"], r["head_end"], r["head_type"],
             r["tail_start"], r["tail_end"], r["tail_type"], r["label"])
            for r in preds
        )
        gold_set = set(
            (r["head_start"], r["head_end"], r["head_type"],
             r["tail_start"], r["tail_end"], r["tail_type"], r["label"])
            for r in golds
        )

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_score(precision: float, recall: float, f1: float) -> float:
    """计算评分公式：Score = 0.5*F1 + 0.25*P + 0.25*R"""
    return 0.5 * f1 + 0.25 * precision + 0.25 * recall


def compute_total_score(ner_metrics: Dict, re_metrics: Dict) -> float:
    """计算最终总分：Total = 0.4*NER_Score + 0.6*RE_Score"""
    ner_score = compute_score(ner_metrics["precision"], ner_metrics["recall"], ner_metrics["f1"])
    re_score = compute_score(re_metrics["precision"], re_metrics["recall"], re_metrics["f1"])
    return 0.4 * ner_score + 0.6 * re_score
