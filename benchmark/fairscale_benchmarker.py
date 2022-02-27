from .base_benchmarker import Benchmaker
from fairscale.nn import ShardedDataParallel, FullyShardedDataParallel
from fairscale.optim import OSS
from torch.nn.parallel import DistributedDataParallel
from colossalai.context import Config
from typing import List
from torch import Tensor
import torch


class FairScaleBenchmarker(Benchmaker):

    def __init__(self, config_path, optim_class, model, criterion):
        config = Config.from_file(config_path)

        if config.stage == 1:
            optimizer = OSS(model.parameters(), optim_class, **config.optimizer, **config.zero.optimizer)
            model = DistributedDataParallel(model)
        elif config.stage == 2:
            optimizer = OSS(model.parameters(), optim_class, **config.optimizer, **config.zero.optimizer)
            model = ShardedDataParallel(model, sharded_optimizer=optimizer, **config.zero.model)
        else:
            if hasattr(config, 'offload') and config.offload:
                model = model.cpu()
            model = FullyShardedDataParallel(model, **config.zero.model)
            optimizer = optim_class(model.parameters(), **config.zero.optimizer)

        if hasattr(config, 'autocast'):
            self.autocast = True
        else:
            self.autocast = False

        super().__init__(config, optimizer, model, criterion)

    def run_iter(self, data: List[Tensor], label: List[Tensor]):
        self.optimizer.zero_grad()

        if self.config.stage == 3:
            self.model.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.autocast):
            out = self.model(*data)
            loss = self.criterion(out, *label)
        loss.backward()
        self.optimizer.step()
