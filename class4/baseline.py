"""
联合命名实体识别(NER) + 关系抽取(RE) 基线模型
使用 BERT + 双任务头，支持 Apple MPS 加速
"""

import json
import os
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall
from seqeval.scheme import IOB2

# ========================
# 配置
# ========================
class Config:
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"  # 生物医学BERT
    # model_name = "bert-base-uncased"  # 备选通用BERT
    max_seq_len = 512
    batch_size = 8
    epochs = 20
    lr = 2e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    dropout = 0.1
    ner_loss_weight = 1.0
    re_loss_weight = 1.0
    max_entity_pairs = 64  # 每个样本最多采样多少实体对用于RE
    negative_ratio = 3.0  # RE负样本:正样本比例
    seed = 42
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_split = 0.85  # 85%训练, 15%验证
    data_path = "/Users/ray/Documents/Fork/DeepLearning/class4/HW_train_data.json"
    output_dir = "/Users/ray/Documents/Fork/DeepLearning/class4/output"
    model_save_path = "/Users/ray/Documents/Fork/DeepLearning/class4/output/best_model.pt"

cfg = Config()

os.makedirs(cfg.output_dir, exist_ok=True)

# 固定随机种子
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device.type == "mps":
    torch.mps.manual_seed(cfg.seed)

# ========================
# 标签定义
# ========================
NER_LABELS = ["ABS", "BIS", "BM", "CHR", "CROP", "CROSS", "GENE", "GST", "MRK", "QTL", "TRT", "VAR"]
RE_LABELS = ["AFF", "CON", "HAS", "LOI", "OCI", "USE"]

# BIO标签: O + B-XXX + I-XXX
BIO_LABELS = ["O"]
for label in NER_LABELS:
    BIO_LABELS.append(f"B-{label}")
    BIO_LABELS.append(f"I-{label}")

label2id = {l: i for i, l in enumerate(BIO_LABELS)}
id2label = {i: l for l, i in label2id.items()}
rel2id = {l: i for i, l in enumerate(RE_LABELS)}
id2rel = {i: l for l, i in rel2id.items()}
ner_label_to_bio = {l: f"B-{l}" for l in NER_LABELS}

num_ner_labels = len(BIO_LABELS)
num_rel_labels = len(RE_LABELS)

print(f"Device: {cfg.device}")
print(f"NER labels: {num_ner_labels}, RE labels: {num_rel_labels}")
print(f"Model: {cfg.model_name}")


# ========================
# 数据加载
# ========================
def load_data(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)


def convert_to_bio(text: str, entities: List[Dict]) -> List[str]:
    """将实体列表转换为BIO标签序列 (字符级别)"""
    labels = ["O"] * len(text)
    # 按start排序
    sorted_ents = sorted(entities, key=lambda e: (e["start"], e["end"]))
    for ent in sorted_ents:
        label = ent["label"]
        if label not in NER_LABELS:
            continue
        for i in range(ent["start"], ent["end"]):
            if i == ent["start"]:
                labels[i] = f"B-{label}"
            else:
                labels[i] = f"I-{label}"
    return labels


def align_bio_to_tokens(text: str, bio_labels: List[str], tokenizer, max_len: int):
    """
    将字符级BIO标签对齐到BERT的subword token
    使用-100标记special tokens和padding
    """
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
        return_token_type_ids=False,
    )

    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    offset_mapping = encoding["offset_mapping"][0]

    token_labels = [-100] * len(input_ids)  # 默认ignore

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:  # special token
            continue
        if start < len(bio_labels):
            token_labels[i] = label2id.get(bio_labels[start], label2id["O"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(token_labels, dtype=torch.long),
        "offset_mapping": offset_mapping,
    }


def create_re_samples(
    text: str, entities: List[Dict], relations: List[Dict]
) -> Tuple[List[Tuple[int, int, int, int, int]], int]:
    """
    为关系抽取构造正负样本
    返回: [(head_start, head_end, tail_start, tail_end, rel_id), ...] 和 正样本数
    """
    # 构建gold关系映射
    gold_relations = {}  # (head_start, head_end, tail_start, tail_end) -> rel_id
    for r in relations:
        key = (r["head_start"], r["head_end"], r["tail_start"], r["tail_end"])
        gold_relations[key] = rel2id[r["label"]]

    # 收集所有可能的实体对
    entity_spans = [(e["start"], e["end"], e["label"]) for e in entities]

    positive_pairs = []
    negative_pairs = []

    for i, (h_start, h_end, h_label) in enumerate(entity_spans):
        for j, (t_start, t_end, t_label) in enumerate(entity_spans):
            if i == j:
                continue
            key = (h_start, h_end, t_start, t_end)
            if key in gold_relations:
                positive_pairs.append((h_start, h_end, t_start, t_end, gold_relations[key]))
            else:
                negative_pairs.append((h_start, h_end, t_start, t_end, -1))

    # 采样
    n_pos = len(positive_pairs)
    if n_pos == 0:
        n_neg = min(len(negative_pairs), cfg.max_entity_pairs)
        sampled_neg = random.sample(negative_pairs, n_neg) if negative_pairs else []
        return positive_pairs + [(h, he, t, te, num_rel_labels) for h, he, t, te, _ in sampled_neg], 0

    n_neg_target = min(n_pos * cfg.negative_ratio, cfg.max_entity_pairs - n_pos)
    n_neg_target = max(0, int(n_neg_target))
    sampled_neg = random.sample(negative_pairs, min(n_neg_target, len(negative_pairs))) if negative_pairs else []

    all_pairs = positive_pairs + [(h, he, t, te, num_rel_labels) for h, he, t, te, _ in sampled_neg]
    random.shuffle(all_pairs)
    all_pairs = all_pairs[:cfg.max_entity_pairs]

    return all_pairs, n_pos


def align_entity_spans_to_tokens(entity_spans: List[Tuple[int, int]], offset_mapping):
    """将字符级别的实体span映射到token级别的span"""
    token_spans = []
    char_to_token = {}
    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == char_end:
            continue
        for c in range(char_start, char_end):
            char_to_token[c] = token_idx

    for char_start, char_end in entity_spans:
        if char_start in char_to_token and (char_end - 1) in char_to_token:
            t_start = char_to_token[char_start]
            t_end = char_to_token[char_end - 1] + 1
            token_spans.append((t_start, t_end))
        else:
            token_spans.append(None)
    return token_spans


# ========================
# Dataset
# ========================
class NER_RE_Dataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len: int, training: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.training = training
        self.samples = []

        for sample in data:
            text = sample["text"]
            entities = sample["entities"]
            relations = sample["relations"]

            # 字符级BIO标签
            bio_labels = convert_to_bio(text, entities)

            # Tokenize并对齐
            encoding = self._tokenize(text)
            token_labels = self._align_bio(encoding["offset_mapping"], bio_labels)

            # 构建实体token span映射
            entity_char_spans = [(e["start"], e["end"]) for e in entities]
            entity_token_spans = align_entity_spans_to_tokens(
                entity_char_spans, encoding["offset_mapping"]
            )

            # 构建RE样本
            re_pairs, n_pos = create_re_samples(text, entities, relations)
            # 将字符span转为token span用于RE
            token_re_pairs = []
            for hc, he, tc, te, rid in re_pairs:
                h_ts = align_entity_spans_to_tokens([(hc, he)], encoding["offset_mapping"])[0]
                t_ts = align_entity_spans_to_tokens([(tc, te)], encoding["offset_mapping"])[0]
                if h_ts is not None and t_ts is not None:
                    token_re_pairs.append((h_ts[0], h_ts[1], t_ts[0], t_ts[1], rid))
                # 如果无法对齐（被截断），跳过该对

            self.samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "ner_labels": torch.tensor(token_labels, dtype=torch.long),
                "re_pairs": token_re_pairs,
                "entity_token_spans": entity_token_spans,
                "entity_types": [ner_label_to_bio.get(e["label"], "O") for e in entities],
                "has_re": n_pos > 0,
            })

    def _tokenize(self, text: str):
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "offset_mapping": encoding["offset_mapping"][0],
        }

    def _align_bio(self, offset_mapping, bio_labels: List[str]):
        token_labels = [-100] * len(offset_mapping)
        for i, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue
            if start < len(bio_labels):
                token_labels[i] = label2id.get(bio_labels[start], label2id["O"])
        return token_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "attention_mask": s["attention_mask"],
            "ner_labels": s["ner_labels"],
            "re_pairs": s["re_pairs"],
            "idx": idx,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    ner_labels = torch.stack([b["ner_labels"] for b in batch])

    # re_pairs: list of lists, 每个样本的pairs数量不同
    re_pairs = [b["re_pairs"] for b in batch]
    idxs = [b["idx"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "re_pairs": re_pairs,
        "idxs": idxs,
    }


# ========================
# 模型
# ========================
class JointNERREModel(nn.Module):
    def __init__(self, model_name: str, num_ner_labels: int, num_rel_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_rel_labels = num_rel_labels

        # NER头: token级别分类
        self.ner_classifier = nn.Linear(self.hidden_size, num_ner_labels)
        self.ner_dropout = nn.Dropout(dropout)

        # RE头: 实体对分类
        # head_emb, tail_emb, head_emb*tail_emb (Hadamard积), head_type, tail_type
        # 类型嵌入: 每种NER类型一个嵌入
        self.type_embedding = nn.Embedding(num_ner_labels, 64)
        rel_input_dim = self.hidden_size * 3 + 64 * 2  # concat + hadamard + type_embs
        self.rel_classifier = nn.Sequential(
            nn.Linear(rel_input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels + 1),  # +1 for "no_relation"
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        # NER logits
        ner_logits = self.ner_classifier(self.ner_dropout(sequence_output))  # (B, L, num_ner_labels)

        return sequence_output, ner_logits

    def predict_relations(self, sequence_output, re_pairs, ner_logits):
        """
        对实体对进行关系分类
        re_pairs: list of list of (h_start, h_end, t_start, t_end, rel_id)
        返回每个pair的logits
        """
        device = sequence_output.device
        batch_size = len(re_pairs)
        all_logits = []
        all_labels = []

        for b in range(batch_size):
            if len(re_pairs[b]) == 0:
                all_logits.append(torch.empty(0, self.num_rel_labels + 1, device=device))
                all_labels.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            batch_logits = []
            batch_labels = []
            seq_out = sequence_output[b]  # (L, H)
            ner_pred = ner_logits[b].argmax(dim=-1)  # (L,)

            for h_start, h_end, t_start, t_end, rel_id in re_pairs[b]:
                # 池化实体表示 (mean pooling)
                head_emb = seq_out[h_start:h_end].mean(dim=0)  # (H,)
                tail_emb = seq_out[t_start:t_end].mean(dim=0)  # (H,)

                # 类型嵌入 (MPS不支持mode, 手动计算众数)
                if h_end > h_start:
                    h_span_preds = ner_pred[h_start:h_end].cpu()
                    h_type_id = torch.bincount(h_span_preds).argmax().item()
                else:
                    h_type_id = 0
                if t_end > t_start:
                    t_span_preds = ner_pred[t_start:t_end].cpu()
                    t_type_id = torch.bincount(t_span_preds).argmax().item()
                else:
                    t_type_id = 0
                h_type_id = min(h_type_id, self.type_embedding.num_embeddings - 1)
                t_type_id = min(t_type_id, self.type_embedding.num_embeddings - 1)
                h_type_emb = self.type_embedding(torch.tensor(h_type_id, device=device))
                t_type_emb = self.type_embedding(torch.tensor(t_type_id, device=device))

                # 拼接特征
                pair_features = torch.cat([
                    head_emb,
                    tail_emb,
                    head_emb * tail_emb,
                    h_type_emb,
                    t_type_emb,
                ])  # (H*3 + 128,)

                logits = self.rel_classifier(pair_features)  # (num_rel_labels+1,)
                batch_logits.append(logits)
                batch_labels.append(rel_id)

            if batch_logits:
                all_logits.append(torch.stack(batch_logits))
                all_labels.append(torch.tensor(batch_labels, dtype=torch.long, device=device))
            else:
                all_logits.append(torch.empty(0, self.num_rel_labels + 1, device=device))
                all_labels.append(torch.empty(0, dtype=torch.long, device=device))

        return all_logits, all_labels


# ========================
# 损失函数
# ========================
def compute_losses(ner_logits, ner_labels, rel_logits_list, rel_labels_list):
    # NER loss
    ner_loss = nn.CrossEntropyLoss(ignore_index=-100)(
        ner_logits.view(-1, ner_logits.size(-1)),
        ner_labels.view(-1),
    )

    # RE loss
    rel_loss = torch.tensor(0.0, device=ner_logits.device)
    rel_count = 0
    for logits, labels in zip(rel_logits_list, rel_labels_list):
        if len(labels) > 0:
            rel_loss += nn.CrossEntropyLoss()(logits, labels)
            rel_count += 1

    if rel_count > 0:
        rel_loss = rel_loss / rel_count

    total_loss = cfg.ner_loss_weight * ner_loss + cfg.re_loss_weight * rel_loss
    return total_loss, ner_loss, rel_loss


# ========================
# 评估
# ========================
def compute_ner_metrics(true_labels: List[List[str]], pred_labels: List[List[str]]):
    """使用字符级BIO标签计算NER指标"""
    p = seqeval_precision(true_labels, pred_labels, scheme=IOB2)
    r = seqeval_recall(true_labels, pred_labels, scheme=IOB2)
    f1 = seqeval_f1(true_labels, pred_labels, scheme=IOB2)
    score = 0.5 * f1 + 0.25 * p + 0.25 * r
    return p, r, f1, score


def compute_re_metrics(true_relations: List[Dict], pred_relations: List[Dict]):
    """
    计算RE的P/R/F1
    true_relations: [{"head": ..., "tail": ..., "label": ...}, ...]
    pred_relations: 同上
    """
    true_set = set()
    for r in true_relations:
        true_set.add((r["head"], r["tail"], r["label"]))

    pred_set = set()
    for r in pred_relations:
        pred_set.add((r["head"], r["tail"], r["label"]))

    if len(pred_set) == 0:
        p = 0.0
    else:
        p = len(true_set & pred_set) / len(pred_set)

    if len(true_set) == 0:
        r = 0.0
    else:
        r = len(true_set & pred_set) / len(true_set)

    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)

    score = 0.5 * f1 + 0.25 * p + 0.25 * r
    return p, r, f1, score


def decode_entities_from_bio(bio_token_labels, input_ids, tokenizer, offset_mapping):
    """从token级别的BIO预测解码实体"""
    entities = []
    current_entity = None

    for i, (label_id, (start, end)) in enumerate(zip(bio_token_labels, offset_mapping)):
        if start == end:
            continue
        label_str = id2label.get(label_id, "O")

        if label_str.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = label_str[2:]
            current_entity = {"start": start, "end": end, "label": entity_type, "text": None}  # text later
        elif label_str.startswith("I-"):
            if current_entity is not None and current_entity["label"] == label_str[2:]:
                current_entity["end"] = end
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = None
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def predict_and_evaluate(model, dataloader, dataset, tokenizer):
    """完整的评估流程"""
    model.eval()
    all_true_entities = []
    all_pred_entities = []
    all_true_relations = []
    all_pred_relations = []
    all_texts = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            re_pairs = batch["re_pairs"]
            idxs = batch["idxs"]

            sequence_output, ner_logits = model(input_ids, attention_mask)
            ner_preds = ner_logits.argmax(dim=-1).cpu().numpy()  # (B, L)
            rel_logits_list, _ = model.predict_relations(sequence_output, re_pairs, ner_logits)

            for b, idx in enumerate(idxs):
                sample = dataset.data[idx]
                text = sample["text"]
                true_entities = sample["entities"]
                true_relations = sample["relations"]

                # Decode NER predictions -> character level
                offset_map = dataset.samples[idx]["input_ids"].numpy() if False else None
                # We need to re-tokenize to get offset mapping
                encoding = tokenizer(
                    text,
                    max_length=cfg.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    return_token_type_ids=False,
                )
                offset_mapping = encoding["offset_mapping"][0].numpy()

                # Decode entities
                pred_entities_chars = decode_entities_from_bio(
                    ner_preds[b].tolist(), input_ids[b], tokenizer, offset_mapping
                )
                # Fill in text
                for ent in pred_entities_chars:
                    ent["text"] = text[ent["start"]:ent["end"]]

                # Deduplicate and filter
                deduped = []
                seen = set()
                for ent in pred_entities_chars:
                    key = (ent["start"], ent["end"], ent["label"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(ent)
                pred_entities_chars = deduped

                # Predict relations
                pred_rels = []
                if len(rel_logits_list[b]) > 0:
                    rel_preds = rel_logits_list[b].argmax(dim=-1).cpu().numpy()
                    for ri, (h_start, h_end, t_start, t_end, _) in enumerate(re_pairs[b]):
                        pred_rel_id = rel_preds[ri]
                        if pred_rel_id < num_rel_labels:  # not "no_relation"
                            # Map token span back to char span
                            h_char_start = offset_mapping[h_start][0]
                            h_char_end = offset_mapping[min(h_end - 1, len(offset_mapping) - 1)][1]
                            t_char_start = offset_mapping[t_start][0]
                            t_char_end = offset_mapping[min(t_end - 1, len(offset_mapping) - 1)][1]

                            pred_rels.append({
                                "head": text[h_char_start:h_char_end],
                                "tail": text[t_char_start:t_char_end],
                                "label": id2rel[pred_rel_id],
                            })

                # Deduplicate relations
                rel_deduped = []
                rel_seen = set()
                for r in pred_rels:
                    key = (r["head"], r["tail"], r["label"])
                    if key not in rel_seen:
                        rel_seen.add(key)
                        rel_deduped.append(r)

                # Convert true entities to char-level BIO for seqeval
                true_bio = convert_to_bio(text, true_entities)
                pred_bio = ["O"] * len(text)
                for ent in pred_entities_chars:
                    label = ent["label"]
                    if label not in NER_LABELS:
                        continue
                    for i in range(ent["start"], min(ent["end"], len(text))):
                        if i == ent["start"]:
                            pred_bio[i] = f"B-{label}"
                        else:
                            pred_bio[i] = f"I-{label}"

                all_true_entities.append(true_bio)
                all_pred_entities.append(pred_bio)
                all_true_relations.append(true_relations)
                all_pred_relations.append(rel_deduped)
                all_texts.append(text)

    # 计算指标
    ner_p, ner_r, ner_f1, ner_score = compute_ner_metrics(all_true_entities, all_pred_entities)

    # RE: 展平所有关系
    flat_true_rels = []
    flat_pred_rels = []
    for tr, pr in zip(all_true_relations, all_pred_relations):
        for r in tr:
            flat_true_rels.append(r)
        for r in pr:
            flat_pred_rels.append(r)

    re_p, re_r, re_f1, re_score = compute_re_metrics(flat_true_rels, flat_pred_rels)

    total_score = 0.4 * ner_score + 0.6 * re_score

    return {
        "ner_p": ner_p, "ner_r": ner_r, "ner_f1": ner_f1, "ner_score": ner_score,
        "re_p": re_p, "re_r": re_r, "re_f1": re_f1, "re_score": re_score,
        "total_score": total_score,
    }


# ========================
# 训练
# ========================
def train():
    print("Loading data...")
    all_data = load_data(cfg.data_path)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * cfg.train_split)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = NER_RE_Dataset(train_data, tokenizer, cfg.max_seq_len, training=True)
    val_dataset = NER_RE_Dataset(val_data, tokenizer, cfg.max_seq_len, training=False)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print("Initializing model...")
    model = JointNERREModel(cfg.model_name, num_ner_labels, num_rel_labels, cfg.dropout)
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_total_score = 0.0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_loss_sum = 0.0
        ner_loss_sum = 0.0
        re_loss_sum = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            ner_labels = batch["ner_labels"].to(cfg.device)
            re_pairs = batch["re_pairs"]

            sequence_output, ner_logits = model(input_ids, attention_mask)
            rel_logits_list, rel_labels_list = model.predict_relations(
                sequence_output, re_pairs, ner_logits
            )

            total_loss, ner_loss, re_loss = compute_losses(
                ner_logits, ner_labels, rel_logits_list, rel_labels_list
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss_sum += total_loss.item()
            ner_loss_sum += ner_loss.item()
            re_loss_sum += re_loss.item()

        avg_loss = total_loss_sum / len(train_loader)
        avg_ner = ner_loss_sum / len(train_loader)
        avg_re = re_loss_sum / len(train_loader)

        # 验证
        metrics = predict_and_evaluate(model, val_loader, val_dataset, tokenizer)

        print(f"Epoch {epoch+1}/{cfg.epochs} | "
              f"Loss: {avg_loss:.4f} (NER:{avg_ner:.4f} RE:{avg_re:.4f}) | "
              f"NER: P={metrics['ner_p']:.4f} R={metrics['ner_r']:.4f} F1={metrics['ner_f1']:.4f} S={metrics['ner_score']:.4f} | "
              f"RE: P={metrics['re_p']:.4f} R={metrics['re_r']:.4f} F1={metrics['re_f1']:.4f} S={metrics['re_score']:.4f} | "
              f"Total: {metrics['total_score']:.4f}")

        if metrics["total_score"] > best_total_score:
            best_total_score = metrics["total_score"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }, cfg.model_save_path)
            print(f"  -> Best model saved! (Total: {best_total_score:.4f})")

    print(f"\nTraining complete! Best epoch: {best_epoch}, Best Total Score: {best_total_score:.4f}")

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = predict_and_evaluate(model, val_loader, val_dataset, tokenizer)
    print(f"\n=== Final Evaluation (Best Model) ===")
    print(f"NER: P={final_metrics['ner_p']:.4f} R={final_metrics['ner_r']:.4f} F1={final_metrics['ner_f1']:.4f} Score={final_metrics['ner_score']:.4f}")
    print(f"RE:  P={final_metrics['re_p']:.4f} R={final_metrics['re_r']:.4f} F1={final_metrics['re_f1']:.4f} Score={final_metrics['re_score']:.4f}")
    print(f"Total Score: {final_metrics['total_score']:.4f}")

    return model, tokenizer


if __name__ == "__main__":
    train()