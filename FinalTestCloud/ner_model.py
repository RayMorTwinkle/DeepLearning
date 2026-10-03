"""
FinalTest - NER 模型定义
    DeBERTa/BERT + Linear + CRF
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizer
from torchcrf import CRF


class NERModel(nn.Module):
    """
    命名实体识别模型
    Encoder (DeBERTa/BERT) → Linear → CRF
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
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
        self.crf = CRF(num_labels, batch_first=True)

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
            labels: [batch, seq_len]  BIOES 标签 ID，-100 表示忽略

        Returns:
            loss: scalar tensor (if labels provided)
            logits: [batch, seq_len, num_labels] (if labels not provided)
        """
        # 确保输入和模型权重类型一致
        input_dtype = next(self.encoder.parameters()).dtype
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 确保 hidden_states 和 classifier 权重类型一致
        if hidden_states.dtype != self.classifier.weight.dtype:
            hidden_states = hidden_states.to(self.classifier.weight.dtype)

        hidden_states = self.dropout(hidden_states)
        emissions = self.classifier(hidden_states)  # [batch, seq_len, num_labels]

        if labels is not None:
            # 替换 -100 为 0（CRF 需要有效标签），并用 mask 忽略
            mask = attention_mask.bool()
            # CRF 的标签必须是有效索引，-100 替换为 0
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction="mean")
            return {"loss": loss, "logits": emissions}
        else:
            return {"logits": emissions}

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Viterbi 解码

        Returns:
            predictions: [batch, seq_len]  解码后的标签 ID
        """
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
            
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 确保类型一致
        if hidden_states.dtype != self.classifier.weight.dtype:
            hidden_states = hidden_states.to(self.classifier.weight.dtype)
            
        emissions = self.classifier(hidden_states)
        mask = attention_mask.bool()
        predictions = self.crf.decode(emissions, mask=mask)
        return predictions


def get_device() -> torch.device:
    """自动检测最优设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
