from __future__ import annotations

import abc
from typing import List

import algorithms as al


class StopCriteria(abc.ABC):

    @abc.abstractmethod
    def isFinished(self, alg: al.Algorithm) -> bool:
        pass


class CompositeStopCriteria(StopCriteria):
    def __init__(self):
        self.children: List[StopCriteria] = []

    def add(self, stop_criteria: StopCriteria):
        self.children.append(stop_criteria)

    def isFinished(self, alg: al.Algorithm) -> bool:
        return any(child.isFinished(alg) for child in self.children)


class MaxEpochStopCriteria(StopCriteria):
    def __init__(self, max_epoch: int = 10):
        self.max_epoch = max_epoch

    def isFinished(self, alg: al.Algorithm) -> bool:
        epoch = getattr(alg, "epoch", 0)
        return epoch >= self.max_epoch


class MinLossStopCriteria(StopCriteria):
    def __init__(self, min_loss: float = 0.1):
        self.min_loss = min_loss

    def isFinished(self, alg: al.Algorithm) -> bool:
        train_loss = getattr(alg, "train_loss", None)
        if train_loss is None:
            return False
        return train_loss <= self.min_loss
