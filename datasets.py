from typing import List, Tuple, cast

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CodeDataset(Dataset):
    def __init__(
        self,
        examples: List[Tuple[str, int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        input_ids_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []
        labels_list: list[int] = []

        for code, label_id in examples:
            enc = tokenizer(
                code,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = cast(torch.Tensor, enc["input_ids"]).squeeze(0)
            attention_mask = cast(
                torch.Tensor, enc["attention_mask"]).squeeze(0)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(label_id)

        self.encodings: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }
        self.labels: torch.Tensor = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }
