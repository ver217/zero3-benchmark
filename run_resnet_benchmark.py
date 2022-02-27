import csv
import torch
import torch.distributed as dist

from torchvision.models import resnet152
from benchmark.utils import parse_args, get_stage, get_autocast_state, get_offload_state
from run_benchmark import run_benchmark


BATCH_SIZE = 64
IMG_SIZE = 224
NUM_CLASS = 1000

WARMUP = 10
TEST_ITERS = 100

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
        with open(f'{args.type}_resnset.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    'Method', 'Stage', 'Autocast', 'Offload', 'Avg time', 'Max mem allocated', 'Max mem cached', 
                    'BParams'
                ])                
            
            autocast = get_autocast_state(args)
            offload = get_offload_state(args)
            stage = get_stage(args)

            writer.writerow([
                args.type,
                str(stage),
                str(autocast),
                str(offload), f'{runtime_stats[0].item():.3f}', f'{runtime_stats[1].item():.2f}',
                f'{runtime_stats[2].item():.2f}',
                f'{model_size:.3f}'
            ])


if __name__ == '__main__':
    main()
