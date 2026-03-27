from tqdm import tqdm
from stop_criteria import StopCriteria
import torch.nn as nn
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
        device,
        lr_encoder: float = 1e-5,
        lr_classifier: float = 1e-3,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        self.lr_encoder = lr_encoder
        self.lr_classifier = lr_classifier
        self.freeze_encoder = freeze_encoder

        self.train_loss: float = 0.0
        self.train_acc: float = 0.0
        self.val_loss: float = 0.0
        self.val_acc: float = 0.0

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
        self.loss_fn = nn.CrossEntropyLoss()

    def _process_batch(self, batch, is_training: bool) -> tuple[float, int, int]:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        if is_training:
            self.optimizer.zero_grad()

        logits = self.model(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

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

    def fit(self, stop_criteria: StopCriteria):

        self.epoch = 0
        self.notify_started()

        while True:

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            self.train_loss = train_loss
            self.train_acc = train_acc
            self.val_loss = val_loss
            self.val_acc = val_acc

            self.epoch += 1
            self.notify_iteration()

            if stop_criteria.isFinished(self):
                break

        self.notify_finished()
