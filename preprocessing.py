import os
import random
from typing import List

from shared_types import CodeExamples

LABELS: List[str] = ["singleton", "factory", "observer", "strategy", "none"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


class Preprocessing:

    @staticmethod
    def load_examples(data_dir: str) -> CodeExamples:

        examples: CodeExamples = []

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

    @staticmethod
    def split_examples(
        examples: CodeExamples,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> tuple[CodeExamples, CodeExamples]:

        random.seed(seed)

        by_label: dict[int, CodeExamples] = {}
        for code, label_id in examples:
            by_label.setdefault(label_id, []).append((code, label_id))

        train_examples: CodeExamples = []
        val_examples: CodeExamples = []

        for items in by_label.values():
            items_shuffled = items[:]
            random.shuffle(items_shuffled)

            n_total = len(items_shuffled)
            n_val = max(1 if n_total > 1 else 0,
                        int(round(n_total * val_ratio)))
            n_val = min(n_val, n_total)

            val_examples.extend(items_shuffled[:n_val])
            train_examples.extend(items_shuffled[n_val:])

        random.shuffle(train_examples)
        random.shuffle(val_examples)

        return train_examples, val_examples
