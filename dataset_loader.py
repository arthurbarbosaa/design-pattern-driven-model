"""
dataset_loader.py — Carregamento e tokenização do dataset de padrões de projeto.

Este módulo:
  1. Percorre as subpastas de `data_dir` (cada subpasta = uma classe/label).
  2. Lê o conteúdo de cada arquivo `.py` como string.
  3. Constrói um `torch.utils.data.Dataset` que tokeniza o código fonte
     usando o tokenizer do CodeBERT (`microsoft/codebert-base`).

Uso típico:
    from dataset_loader import load_examples, CodeDataset, LABEL2ID
    examples = load_examples("dataset")
    dataset  = CodeDataset(examples, tokenizer, max_length=512)
"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# ==================== Mapeamento de labels ====================
# A ordem aqui define o ID numérico de cada classe.
LABELS: List[str] = ["singleton", "factory", "observer", "strategy", "none"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


# ==================== Leitura dos arquivos ====================

def load_examples(data_dir: str) -> List[Tuple[str, int]]:
    """
    Percorre `data_dir/` e retorna uma lista de tuplas (code_string, label_id).

    Estrutura esperada:
        data_dir/
        ├── singleton/
        │   ├── example1.py
        │   └── ...
        ├── factory/
        └── ...

    Cada subpasta deve ter o nome de uma label definida em LABELS.
    """
    examples: List[Tuple[str, int]] = []

    for label_name in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_name)

        # Ignora arquivos soltos e labels desconhecidas
        if not os.path.isdir(label_path):
            continue
        if label_name not in LABEL2ID:
            print(
                f"[AVISO] Pasta '{label_name}' não corresponde a nenhuma label conhecida. Ignorando.")
            continue

        label_id = LABEL2ID[label_name]

        for filename in sorted(os.listdir(label_path)):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(label_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
            examples.append((code, label_id))

    print(f"[INFO] {len(examples)} exemplos carregados de '{data_dir}' "
          f"({', '.join(f'{l}: {sum(1 for _, lid in examples if lid == i)}' for l, i in LABEL2ID.items())})")
    return examples


# ==================== Dataset PyTorch ====================

class CodeDataset(Dataset):
    """
    Dataset PyTorch que armazena código já tokenizado.

    Parâmetros:
        examples   : lista de (code_string, label_id)
        tokenizer  : tokenizer do CodeBERT
        max_length : comprimento máximo de tokens (padrão: 512)
    """

    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.labels: List[int] = []
        self.encodings = {"input_ids": [], "attention_mask": []}

        for code, label_id in examples:
            # Tokeniza o código-fonte usando o tokenizer do CodeBERT
            enc = tokenizer(
                code,
                # se for menor, completa com padding até max_length.
                max_length=max_length,
                # se o código passar do limite, corta no max_length.
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings["input_ids"].append(enc["input_ids"].squeeze(0))
            self.encodings["attention_mask"].append(
                enc["attention_mask"].squeeze(0))
            self.labels.append(label_id)

        # Empilha tensores para acesso indexado eficiente
        self.encodings["input_ids"] = torch.stack(self.encodings["input_ids"])
        self.encodings["attention_mask"] = torch.stack(
            self.encodings["attention_mask"])
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }
