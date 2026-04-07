import abc
import os
from typing import Any
import matplotlib.pyplot as plt

import torch
import algorithms as al
from labels import LABELS
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
        print(f"\n{'='*60}")
        print(f"  Iniciando treinamento")
        print(f"{'='*60}\n")

    def notify_finished(self, alg: al.Algorithm):
        print(f"\n{'='*60}")
        print(f"  Treinamento concluído")
        print(f"{'='*60}\n")

    def notify_iteration(self, alg: al.Algorithm):
        epoch = getattr(alg, "epoch", None)
        train_loss = getattr(alg, "train_loss", None)
        train_acc = getattr(alg, "train_acc", None)

        print(f"Época {epoch}")
        print(f"  Treino  — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print()


class ModelCheckpointObserver(AlgorithmObserver):

    def __init__(self, output_dir: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.best_train_acc = 0.0

    def notify_started(self, alg: al.Algorithm):
        pass

    def notify_finished(self, alg: al.Algorithm):
        pass

    def notify_iteration(self, alg: al.Algorithm):
        train_acc = getattr(alg, "train_acc", None)
        if train_acc is None:
            return

        if train_acc >= self.best_train_acc:
            self.best_train_acc = train_acc
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


class PlotterAlgorithmsObserver(AlgorithmObserver):
    def __init__(self):
        super().__init__()
        self.epochs: list[int] = []
        self.errors: list[float] = []

    def notify_started(self, alg: al.Algorithm):
        self.epochs = []
        self.errors = []

    def notify_finished(self, alg: al.Algorithm):
        pass

    def notify_iteration(self, alg: al.Algorithm):
        epoch = getattr(alg, "epoch", None)
        train_loss = getattr(alg, "train_loss", None)

        if epoch is None or train_loss is None:
            return

        self.epochs.append(int(epoch))
        self.errors.append(float(train_loss))

    def plot(self):
        if not self.epochs:
            print("No data to plot. Run the algorithm first.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs, self.errors, marker="o", linewidth=2)
        plt.title("Erro de Treino por Iteracao")
        plt.xlabel("Iteracao (epoca)")
        plt.ylabel("Erro (train_loss)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
