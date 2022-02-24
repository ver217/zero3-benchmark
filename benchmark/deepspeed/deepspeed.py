import deepspeed
from ..configs.common import *
from models import VOCAB_SIZE


def run_iter(model_engine, optimizer, criterion):
    assert optimizer is None
    # img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda().half()
    # label = torch.randint(0, NUM_CLASS, (BATCH_SIZE, )).cuda()
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).cuda()
    mask = torch.ones((BATCH_SIZE, SEQ_LEN), dtype=torch.int64, device=torch.cuda.current_device())

    model_engine.zero_grad()
    # out = model_engine(img)
    out = model_engine(input_ids, mask)
    # loss = criterion(out, label)
    loss = criterion(out, input_ids)
    model_engine.backward(loss)
    model_engine.step()


def init(model, args):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters)
    return model_engine
