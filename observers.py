import abc
import os
from typing import Any

import torch
import algorithms as al
from preprocessing import LABELS
from transformers import PreTrainedTokenizer


class AlgorithmObserver(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def notify_started(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_finished(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_iteration(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_better_valadation_accurency(self, alg: al.Algorithm):
        pass


class ModelCheckpointObserver(AlgorithmObserver):

    def __init__(self, output_dir: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def notify_started(self, alg: al.Algorithm):
        pass

    def notify_finished(self, alg: al.Algorithm):
        pass

    def notify_iteration(self, alg: al.Algorithm):
        pass

    def notify_better_valadation_accurency(self, alg: al.Algorithm):

        model = getattr(alg, "model", None)
        if model is None:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(self.output_dir, "model_state_dict.pt"),
        )

        self.tokenizer.save_pretrained(self.output_dir)

        with open(os.path.join(self.output_dir, "labels.txt"), "w") as f:
            for label in LABELS:
                f.write(label + "\n")
