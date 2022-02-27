from abc import ABC, abstractmethod
from torch import Tensor
from typing import List


class Benchmaker(ABC):

    def __init__(self, config, optimizer, model, criterion):
        self.config = config
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion

    @abstractmethod
    def run_iter(self, data: List[Tensor], label: List[Tensor]):
        pass
