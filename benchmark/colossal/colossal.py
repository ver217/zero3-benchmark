import torch
from colossalai.zero.zero_stage3_develop import ZeroRedundancyLevel3Model
from models import VOCAB_SIZE

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
    offload_config = {}
    if offload:
        model = model.cpu()
        offload_config['device'] = 'cpu'
    model = ZeroRedundancyLevel3Model(model, mixed_precision=True,
                                      reshard_after_forward=False, offload_config=offload_config)
    optimizer = OPTIM(model.parameters(), lr=0.001)
    return model, optimizer
