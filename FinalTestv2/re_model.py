"""
FinalTest - RE 模型定义
    DeBERTa/BERT + Entity Markers + [CLS] 分类
    支持 Focal Loss 处理类别不平衡
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedTokenizer


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    gamma=2 让模型更关注难分样本（如稀有关系类）
    """
    # 【讲解重点】关系类别不均衡很明显，NONE 和常见关系多，USE/OCI 少。
    # Focal Loss 会降低简单样本的权重，让模型更关注难分和稀有关系。
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 可选每类权重
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class REModel(nn.Module):
    """
    关系抽取模型
    输入：带实体标记的文本 [E1]head[/E1] ... [E2]tail[/E2]
    输出：[CLS] token 过 Linear 做关系分类
    """
    # 【讲解重点】RE 模型结构：
    # 1. 输入是加了 [E1]/[E2] 标记的文本
    # 2. DeBERTa/BERT encoder 理解这对实体在句中的关系
    # 3. 取 [CLS] 向量做 7 类分类：NONE + 6 种关系

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        init_from_config: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels

        if init_from_config:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 损失函数：Focal Loss 处理类别不平衡，或标准 CrossEntropy
        if use_focal_loss:
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch]  关系标签 ID

        Returns:
            loss: scalar tensor
            logits: [batch, num_labels]
        """
        # 确保输入是 long 类型
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
            
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 【讲解重点】用 [CLS] 位置表示整句话和目标实体对的信息。
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        # 确保 hidden_states 和 classifier 权重类型一致
        if cls_hidden.dtype != self.classifier.weight.dtype:
            cls_hidden = cls_hidden.to(self.classifier.weight.dtype)

        cls_hidden = self.dropout(cls_hidden)
        # 【讲解重点】logits 是每种关系类型的打分。
        logits = self.classifier(cls_hidden)  # [batch, num_labels]

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        推理：返回预测标签和概率

        Returns:
            pred_labels: [batch]
            probs: [batch, num_labels]
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
            
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        
        # 确保类型一致
        if cls_hidden.dtype != self.classifier.weight.dtype:
            cls_hidden = cls_hidden.to(self.classifier.weight.dtype)
            
        logits = self.classifier(cls_hidden)
        # 【讲解重点】推理时转成概率，后面 predict.py 会用阈值过滤低置信度关系。
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(logits, dim=-1)
        return pred_labels, probs
