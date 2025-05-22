from __future__ import print_function
import os
import time
import utils
import wandb
from einops import rearrange
import torch
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from sparselearning.core import Masking, CosineDecay
from sparselearning.resnet_imagenet import ResNet50_imagenet
from sparselearning.resnet_wr import WideResNet_28
from sparselearning.sparse_utils import accuracy_imagenet,get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, get_train_loader_imagenet, get_val_loader_imagenet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def prefetched_loader(loader):
    """Prefetches data to GPU memory."""
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

    stream = torch.cuda.Stream()
    first = True
    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def to_python_float(t):
    """Converts a tensor to a Python float."""
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    """Trains the model for one epoch."""
    model.train()
    train_loss = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_m = AverageMeter()

    for batch_idx, (data, target) in enumerate(prefetched_loader(train_loader)):
        for repetition in range(args.batch_repetition):
            loss = 0
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            optimizer.zero_grad()

            output = model(data)

            if args.num_ensemble > 1:
                output = torch.transpose(output, 0, 1)
                loss = utils.default_loss(output, target)
                loss = loss / output.size()[0]
                output = output.mean(dim=0)
            else:
                loss = utils.default_loss_single(output, target)

            prec1, prec5 = accuracy_imagenet(output, target, topk=(1, 5))

            top1.update(to_python_float(prec1), output.size(0))
            top5.update(to_python_float(prec5), output.size(0))
            loss_m.update(to_python_float(loss), output.size(0))

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            train_loss += loss.item()

            if mask is not None:
                mask.step()
            else:
                optimizer.step()

            torch.cuda.synchronize()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                           100. * batch_idx / len(train_loader), loss.item()))

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        metrics = {'epoch': epoch, 'train_acc': top1.avg, "train_top5": top5.avg, 'train_loss': loss_m.avg}
        wandb.log(metrics)
    print('\n{}: Average loss: {:.4f}\n'.format('Training summary', train_loss / batch_idx))


def evaluate(args, model, device, test_loader, is_test_set=False):
    """Evaluates the current model."""
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_m = AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(prefetched_loader(test_loader)):
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)

            if args.num_ensemble > 1:
                output = torch.transpose(output, 0, 1)
                loss = utils.default_loss(output, target)
                loss = loss / output.size()[0]
                output = output.mean(dim=0)
            else:
                loss = utils.default_loss_single(output, target)

            prec1, prec5 = accuracy_imagenet(output, target, topk=(1, 5))

            top1.update(to_python_float(prec1), output.size(0))
            top5.update(to_python_float(prec5), output.size(0))
            loss_m.update(to_python_float(loss), output.size(0))

    print('\nAverage loss: {:.4f}, Accuracy1 and 5: {} and {}'.format(
        loss_m.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg, loss_m.avg


def main():
    start_train_time = time.time()
    args = utils.parse_args()
    args.distributed = False
    #args.data_location = os.environ['IMAGENET_PYTORCH']

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1


    if args.distributed:
        os.environ['MASTER_PORT'] = str(args.mst_prt)
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu = args.local_rank % torch.cuda.device_count()
        print(args.gpu, 'gpu')

        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    utils.print_args(args)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.init(project="neurotrails", config=vars(args), name=utils.set_experiment_name(args), mode=args.wandb_mode)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    print('\n\n')
    print('=' * 80)
    torch.manual_seed(args.seed)

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                          max_threads=args.max_threads)
    elif args.data == 'cifar100':
        train_loader, test_loader = get_cifar100_dataloaders(args)
    elif args.data == 'imagenet':
        train_loader = get_train_loader_imagenet(args.imagenet_location, args.batch_size, workers=5)
        test_loader = get_val_loader_imagenet(args.imagenet_location, args.batch_size, workers=5)


    if args.model == 'ResNet50':
        model = ResNet50_imagenet(args).to(device)
    elif args.model == 'WideResNet':
        model = WideResNet_28(args).to(device)
    else:
        raise Exception('Model not implemented yet.')

    if args.distributed:
        # model = DDP(model, device_ids=[args.local_rank])
        model = DDP(model)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,
                              nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        print('Unknown optimizer: {0}'.format(args.optimizer))
        raise Exception('Unknown optimizer.')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(args.epochs * 0.25),int(args.epochs * 0.5), int(args.epochs * 0.75)],
                                                        last_epoch=-1)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            original_acc = evaluate(args, model, device, test_loader)

    if args.fp16:
        print('FP16')
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'init_scale': 2 ** 16})
        model = model.half()

    mask = None
    if not args.density == 1:
        
        decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs))
        mask = Masking(optimizer, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                        redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
        mask.add_module(model, sparse_init=args.sparse_init, density=args.density, load_masks = None)

    best_acc = 0.0

    save_path = 'real_ensemble/' + str(args.model)
    save_subfolder = os.path.join(save_path,
                                  str(args.model) + '_base_ens=' + str(args.baseline_ensemble) + '_density' + str(
                                      args.density) + 'num_ensemble' + str(args.num_ensemble) + '_seed' + str(
                                      args.seed) + '_blocks_in_head' + str(args.blocks_in_head) + '_epochs' + str(
                                      args.epochs) + 'update_frequency' + str(args.update_frequency))

    if not os.path.exists(save_subfolder): os.makedirs(save_subfolder,exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if torch.distributed.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch, mask)
        lr_scheduler.step()
        if epoch % 5==0:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

                val_prec1,val_prec5,val_loss = evaluate(args, model, device, test_loader)
                metrics = {'epoch':epoch, "val_acc":val_prec1, "val_top5":val_prec5, "val_loss":val_loss}

                wandb.log(metrics)
                if val_prec1 > best_acc:
                    best_acc = val_prec1

                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'masks': mask,
                        
                    }, filename=os.path.join(save_subfolder, 'model_final.pth'))

        print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
            optimizer.param_groups[0]['lr'], time.time() - t0))
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('Best prec1 is:', best_acc)
        total_training_hours = (time.time() - start_train_time) / 3600.
        print('Total training time (hours) is:', total_training_hours)
        wandb.log({"best_prec1": best_acc, "total_training_hours": total_training_hours})


if __name__ == '__main__':
    main()
