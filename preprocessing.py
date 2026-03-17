import os
from typing import List, Tuple

LABELS: List[str] = ["singleton", "factory", "observer", "strategy", "none"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


class Preprocessing:

    @staticmethod
    def load_examples(data_dir: str) -> List[Tuple[str, int]]:

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
