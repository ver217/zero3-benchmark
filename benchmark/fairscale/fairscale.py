import torch
from models import VOCAB_SIZE

from fairscale.nn import FullyShardedDataParallel

from ..configs.common import *


def run_iter(model, optimizer, criterion, autocast=False):
    # img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda()
    # label = torch.randint(0, NUM_CLASS, (BATCH_SIZE, )).cuda()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).cuda()
    mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.int64, device=torch.cuda.current_device())
    optimizer.zero_grad()
    model.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=autocast):
        # out = model(img)
        # loss = criterion(out, label)
        out = model(input_ids, mask)
        # print(f'out device: {out.device}')
        loss = criterion(out, input_ids)
        # print(f'loss device: {loss.device}')
    loss.backward()
    optimizer.step()


def init(model, offload=False):
    if offload:
        model = model.cpu()
    model = FullyShardedDataParallel(model, mixed_precision=True, flatten_parameters=False,
                                     reshard_after_forward=False, move_params_to_cpu=offload, move_grads_to_cpu=offload)
    optimizer = OPTIM(model.parameters(), lr=0.001)
    return model, optimizer
