from .base_benchmarker import Benchmaker
from colossalai.context import Config
import torch


class ColossalaiBenchmarker(Benchmaker):

    def __init__(self, config_path, optim_class, model, criterion):
        config = Config.from_file(config_path)
        optimizer = optim_class(model.parameters(), **config.optimizer)
        if config.stage in [1, 2]:
            from colossalai.zero.zero_stage2_develop import ShardedOptimizer
            model = model.half().cuda()
            optimizer = ShardedOptimizer(optimizer=optimizer, **config.zero.optimizer)
        elif config.stage == 3:
            from colossalai.zero.zero_stage3_develop import ZeroRedundancyLevel3Model
            offload_config = {}
            if config.offload:
                model = model.cpu()
                offload_config['device'] = 'cpu'
            model = ZeroRedundancyLevel3Model(model=model, **config.zero.model)
        else:
            raise ValueError('invalid stage number')
        super().__init__(config, optimizer, model, criterion)

    def run_iter(self, data, label):
        if self.config.stage in [1, 2]:
            self._run_iter_stage_1_2(data, label)
        elif self.config.stage == 3:
            self._run_iter_stage_3(data, label)

    def _run_iter_stage_1_2(self, data, label):
        data = [val.half() if val.dtype == torch.float else val for val in data]
        self.optimizer.zero_grad()

        # forward
        out = self.model(*data)
        loss = self.criterion(out, *label)

        # backward
        self.optimizer.backward(loss)
        self.optimizer.sync_grad()
        self.optimizer.step()

    def _run_iter_stage_3(self, data, label):
        self.optimizer.zero_grad()
        self.model.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.config.autocast):
            out = self.model(*data)
            if torch.is_tensor(out):
                out = [out]
            loss = self.criterion(*out, *label)
        loss.backward()
        self.optimizer.step()
