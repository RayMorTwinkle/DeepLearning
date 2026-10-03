"""
FinalTest - RE 模型定义
    DeBERTa/BERT + Entity Markers + [CLS] 分类
"""
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedTokenizer


class REModel(nn.Module):
    """
    关系抽取模型
    输入：带实体标记的文本 [E1]head[/E1] ... [E2]tail[/E2]
    输出：[CLS] token 过 Linear 做关系分类
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 损失函数：处理类别不平衡
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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        cls_hidden = self.dropout(cls_hidden)
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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden)
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(logits, dim=-1)
        return pred_labels, probs
