import torch
import deepspeed
from .base_benchmarker import Benchmaker
from typing import List
from torch import Tensor


class DeepSpeedBenchmarker(Benchmaker):

    def __init__(self, ds_args, model, criterion):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_engine, optimizer, _, _ = deepspeed.initialize(args=ds_args, model=model, model_parameters=parameters)

        super().__init__(ds_args, optimizer, model_engine, criterion)

    def run_iter(self, data: List[Tensor], label: List[Tensor]):
        data = [val.half() if val.dtype == torch.float else val for val in data]
        self.model.zero_grad()
        out = self.model(*data)
        loss = self.criterion(out, *label)
        self.model.backward(loss)
        self.model.step()
