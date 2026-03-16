"""
model.py — Definição do modelo de classificação baseado em CodeBERT.

Arquitetura:
    ┌────────────────────┐
    │  CodeBERT encoder  │   (microsoft/codebert-base — 12 camadas Transformer)
    └────────┬───────────┘
             │  hidden state do token [CLS]  (768-d)
    ┌────────▼───────────┐
    │      Dropout       │   (p = 0.3)
    └────────┬───────────┘
    ┌────────▼───────────┐
    │   Linear (768→5)   │   (cabeça de classificação)
    └────────┬───────────┘
             │  logits  (5 classes)

O modelo retorna logits crus; a CrossEntropyLoss do PyTorch já
aplica log-softmax internamente, então NÃO adicionamos softmax aqui.
Na inferência usamos softmax explícito para obter probabilidades.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class CodeBERTClassifier(nn.Module):
    """
    Classificador de padrões de projeto baseado em CodeBERT.

    Parâmetros:
        model_name : identificador HuggingFace do encoder (padrão: codebert-base)
        num_labels : número de classes de saída
        dropout    : taxa de dropout antes da camada linear
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_labels: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Encoder pré-treinado (pesos congelados opcionais — aqui fazemos fine-tuning completo)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Tamanho do vetor oculto produzido pelo encoder (768 para roberta-base / codebert-base)
        hidden_size: int = self.encoder.config.hidden_size

        # Cabeça de classificação
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Entradas:
            input_ids      : (batch, seq_len)  — IDs dos tokens
            attention_mask  : (batch, seq_len) — máscara de atenção (1 = token real, 0 = padding)

        Saída:
            logits : (batch, num_labels) — pontuações brutas para cada classe
        """
        # Obtém os hidden states do encoder
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)

        # Usa o hidden state do token [CLS] (primeiro token) como representação da sequência
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        # Dropout + camada linear
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)  # (batch, num_labels)

        return logits
