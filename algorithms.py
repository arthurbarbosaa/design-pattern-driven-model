from tqdm import tqdm
from typing import Optional
import abc

import torch


class Algorithm(abc.ABC):
    def __init__(self):
        self.algorithm_observers = []

    def add(self, observer):
        if observer not in self.algorithm_observers:
            self.algorithm_observers.append(observer)
        else:
            print('Failed to add: {}'.format(observer))

    def remove(self, observer):
        try:
            self.algorithm_observers.remove(observer)
        except ValueError:
            print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    def notify_better_valadation_accurency(self):
        [o.notify_better_valadation_accurency(
            self) for o in self.algorithm_observers]


class FineTuningAlgorithm(Algorithm):
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        device,
        lr_encoder: float = 1e-5,
        lr_classifier: float = 1e-3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device

        self.lr_encoder = lr_encoder
        self.lr_classifier = lr_classifier
        self.freeze_encoder = freeze_encoder

        self._setup()

    def _setup(self):
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        encoder_params = [
            p for p in self.model.encoder.parameters() if p.requires_grad
        ]

        classifier_params = [
            p for p in self.model.classifier.parameters() if p.requires_grad
        ]

        param_groups = []

        if len(encoder_params) > 0:
            param_groups.append({
                "params": encoder_params,
                "lr": self.lr_encoder,
            })

        if len(classifier_params) > 0:
            param_groups.append({
                "params": classifier_params,
                "lr": self.lr_classifier,
            })

        self.optimizer = torch.optim.AdamW(param_groups)

    def _process_batch(self, batch, is_training: bool) -> tuple[float, int, int]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        if is_training:
            self.optimizer.zero_grad()

        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        if is_training:
            loss.backward()
            self.optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)

        return loss.item() * total, correct, total

    def train_one_epoch(self) -> tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_dataloader:
            batch_loss, batch_correct, batch_total = self._process_batch(
                batch, is_training=True)
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_dataloader, desc="  Valid.", leave=False):
            batch_loss, batch_correct, batch_total = self._process_batch(
                batch, is_training=False)
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

        return total_loss / total, correct / total

    def fit(
        self,
        epochs: int,
    ) -> float:
        print(f"\n{'='*60}")
        print(f"  Iniciando treinamento — {epochs} épocas")
        print(f"{'='*60}\n")

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            print(f"Época {epoch}/{epochs}")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            print(f"  Treino  — Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"  Valid.  — Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                self.notify_better_valadation_accurency()

            print()

        return best_val_acc
