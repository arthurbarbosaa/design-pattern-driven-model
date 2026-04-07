from typing import List

LABELS: List[str] = ["singleton", "factory", "observer", "strategy", "none"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}
