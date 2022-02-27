import torch
import torch.distributed as dist

from torchvision.models import resnet152
from benchmark.utils import parse_args
from run_benchmark import run_benchmark
import pprint

BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CLASS = 1000

WARMUP = 1
TEST_ITERS = 10

OPTIM = torch.optim.Adam


def build_model():
    return resnet152().cuda()


def build_criterion():
    return torch.nn.CrossEntropyLoss()


def get_batch_data():
    img = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).cuda()
    label = torch.randint(low=0, high=1000, size=(BATCH_SIZE,)).cuda()
    return [img], [label]


def main():
    args = parse_args()
    model_size, runtime_stats = run_benchmark(args=args,
                                              model_func=build_model,
                                              loss_func=build_criterion,
                                              batch_data_func=get_batch_data,
                                              warmup_steps=WARMUP,
                                              test_steps=TEST_ITERS,
                                              optim_class=OPTIM)

    if dist.get_rank() == 0:

        with open(f'resnet_{args.type}_log.txt', 'a') as f:
            f.write('===========================\n')
            f.write(pprint.pformat(vars(args)))
            f.write('\n-------------\n')
            f.write(f'type: {args.type}\n')
            f.write(f'average step time: {runtime_stats[0].item():.3f}\n')
            f.write(
                f'max memory allocated: {runtime_stats[1].item():.2f}, max memory cached: {runtime_stats[2].item():.2f}'
            )
            f.write(f'model size: {model_size:.3f}B')
            f.write('===========================\n\n')


if __name__ == '__main__':
    main()
