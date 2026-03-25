import torch
import torch.nn as nn
from transformers import AutoModel


class CodeBERTClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size: int = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)

        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)  # (batch, num_labels)

        return logits
