import argparse
import time
import torch
import torch.nn.functional as F
import sparselearning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=float, default=1.0, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')

    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt', help='path to save the final model')
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=bool, default = False)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--save_features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max_threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--nolr_scheduler', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler - linear or cosine plateau')
    parser.add_argument('--wandb_mode', default='disabled', type=str, help='`online` or `offline` or `disabled`', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--console_output', default='', type=str, help='file path where the console output is saved')

    # for ImageNet
    parser.add_argument('--imagenet_location', default='/data/imagenet', type=str, help='ImageNet data location')
    parser.add_argument('--batch_repetition', type=int, default=1, help='Number of times to repeat the batch. Default: 1.')

    # NeuroTrails settings
    parser.add_argument('--num_ensemble', type=int, default=1, help='How many heads in the NeuroTrailsmodel. Default=1')
    parser.add_argument('--blocks_in_head', type=int, default=0, help='How many backbone blocks to be part of each head. Default=0')
    parser.add_argument('--baseline_ensemble', type=int, default=1, help='Number of classifiers in the baseline full ensemble. Default=1')

    # Sparsity settings
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=1.0, help='The pruning rate / death rate.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ER', help='sparse initialization')
    parser.add_argument('--mix', type=float, default=0.0)
    parser.add_argument('--update_frequency', type=int, default=100, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')

    # Multi-GPU settings
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument("--distributed", type=bool,default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--mst_prt", default=10000, type=int)
    parser.add_argument("--clipping",default=0.0,type=float,help='clipping the gradient')

    args = parser.parse_args()
    return args


def print_args(args):
    """Print the arguments used in the experiment."""
    print("\nArguments used:")
    for key in sorted(vars(args)):  # print in alphabetical order
        value = getattr(args, key)
        print(f"{key:30} {value}")
    print()  # newline


def get_num_classes(dataset):
    """Get the number of classes for the given dataset."""
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    elif dataset == 'tiny_imagenet':
        return 200
    else:
        raise NotImplementedError("Unsupported dataset " + dataset)


def set_experiment_name(args):
    """Set the experiment name based on the arguments."""
    model = args.model
    data = args.data
    num_ensemble = args.num_ensemble
    density = args.density
    seed = args.seed
    blocks_in_head = f"_blocHead{args.blocks_in_head}" if args.blocks_in_head > 0 else ''
    baseline_ensemble = f"_baseEns{args.baseline_ensemble}" if args.baseline_ensemble > 1 else ''
    growth = f"_growth{args.growth}"
    return f"{data}_{model}_seed{seed}_dens{density}_heads{num_ensemble}{blocks_in_head}{baseline_ensemble}{growth}"


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    print(f"SAVING CHECKPOINT TO {filename}")
    torch.save(state, filename)


def default_loss(output, target):
    """The default loss function for multi-head models."""
    loss = 0
    for i in range(output.size()[0]):
        individual_loss = F.cross_entropy(output[i], target)
        loss += individual_loss
    return loss


def default_loss_single(output, target):
    """The default loss function for single head models."""
    return F.cross_entropy(output, target)

