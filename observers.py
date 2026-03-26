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


class TrainingPrintObserver(AlgorithmObserver):
    def notify_started(self, alg: al.Algorithm):
        epochs = getattr(alg, "epochs", None)
        if epochs is None:
            return
        print(f"\n{'='*60}")
        print(f"  Iniciando treinamento — {epochs} épocas")
        print(f"{'='*60}\n")

    def notify_finished(self, alg: al.Algorithm):
        print(f"\n{'='*60}")
        print(f"  Treinamento concluído")
        print(f"{'='*60}\n")

    def notify_iteration(self, alg: al.Algorithm):
        epoch = getattr(alg, "epoch", None)
        epochs = getattr(alg, "epochs", None)
        train_loss = getattr(alg, "train_loss", None)
        train_acc = getattr(alg, "train_acc", None)
        val_loss = getattr(alg, "val_loss", None)
        val_acc = getattr(alg, "val_acc", None)

        if None in (epoch, epochs, train_loss, train_acc, val_loss, val_acc):
            return

        print(f"Época {epoch}/{epochs}")
        print(f"  Treino  — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Valid.  — Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print()


class ModelCheckpointObserver(AlgorithmObserver):

    def __init__(self, output_dir: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_val_acc = 0.0

    def notify_started(self, alg: al.Algorithm):
        pass

    def notify_finished(self, alg: al.Algorithm):
        pass

    def notify_iteration(self, alg: al.Algorithm):
        val_acc = getattr(alg, "val_acc", None)
        if val_acc is None:
            return

        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            self._save_checkpoint(alg)

    def _save_checkpoint(self, alg: al.Algorithm):
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
