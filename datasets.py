import torch
from typing import cast
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from shared_types import CodeExamples


class CodeDataset(Dataset):
    def __init__(
        self,
        examples: CodeExamples,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        labels: list[int] = []

        for code, label in examples:
            encoded = tokenizer(
                code,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids.append(
                cast(torch.Tensor, encoded["input_ids"]).squeeze(0))
            attention_mask.append(
                cast(torch.Tensor, encoded["attention_mask"]).squeeze(0))
            labels.append(label)

        self.encodings: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }
