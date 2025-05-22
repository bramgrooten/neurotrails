from __future__ import print_function
import os
import time
import utils
import wandb
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
from sparselearning.core import Masking, CosineDecay
from sparselearning.resnet_wr import WideResNet_28
from sparselearning.resnet_split import ResNet34, ResNet18, ResNet50
from sparselearning.sparse_utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(args, model_or_models, device, train_loader, optimizer_or_optimizers, epoch, mask_or_masks=None):
    """Trains the model for one epoch."""
    is_ensemble_list = isinstance(model_or_models, list)
    
    if is_ensemble_list:
        # Ensemble of separate models
        models = model_or_models
        optimizers = optimizer_or_optimizers
        masks = mask_or_masks if mask_or_masks is not None else [None] * len(models)
        train_loss = defaultdict(lambda: 0)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()

            for index, model in enumerate(models):
                loss = 0
                models[index].train()
                optimizers[index].zero_grad()

                output = models[index](data)
                loss = utils.default_loss_single(output, target)

                if args.fp16:
                    optimizers[index].backward(loss)
                else:
                    loss.backward()

                train_loss[index] += loss.item()

                if masks[0] is not None:
                    masks[index].step()
                else:
                    optimizers[index].step()

            if batch_idx % args.log_interval == 0:
                if batch_idx == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                        100. * batch_idx / len(train_loader), np.mean(list(train_loss.values()))))
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                        100. * batch_idx / len(train_loader), np.mean(list(train_loss.values())) / batch_idx))
        
        wandb.log({"train_loss": np.mean(list(train_loss.values()))/batch_idx})
        print('\n{}: Average loss: {:.4f}\n'.format('Training summary', np.mean(list(train_loss.values()))/batch_idx))
        return np.mean(list(train_loss.values()))
    
    else:
        # Single model with potential ensemble heads
        model = model_or_models
        optimizer = optimizer_or_optimizers
        mask = mask_or_masks
        
        model.train()
        train_loss = 0
        counter = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = 0
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            
            optimizer.zero_grad()
            output = model(data)  # output size = [batch, #ensemble, #class]

                
            if args.num_ensemble > 1:
                output = torch.transpose(output, 0, 1)
                loss = utils.default_loss(output, target)
                loss = loss / output.size()[0]
            else:
                loss = utils.default_loss_single(output, target)

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            train_loss += loss.item()

            if mask is not None:
                mask.step()
            else:
                optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                    100. * batch_idx / len(train_loader), loss.item()))

        wandb.log({"train_loss": train_loss / batch_idx})
        print('\n{}: Average loss: {:.4f}\n'.format('Training summary', train_loss / batch_idx))
        return train_loss


def evaluate(args, model_or_models, device, test_loader, is_test_set=False):
    """Evaluates the current model."""
    is_ensemble_list = isinstance(model_or_models, list)
    
    if is_ensemble_list:
        # Ensemble of separate models
        models = model_or_models
        test_loss = defaultdict(lambda: 0)
        correct = defaultdict(lambda: 0)
        n = defaultdict(lambda: 0)
        test_loss_single = 0
        correct_single = 0
        n_single = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if args.fp16: data = data.half()
                
                outputs = []
                for model in models:
                    model.eval()
                    outputs.append(model(data))
                
                output = torch.stack(outputs)
                
                for head in range(len(output)):
                    pred = output[head].argmax(dim=1, keepdim=True)
                    correct[head] += pred.eq(target.view_as(pred)).sum().item()
                    n[head] += target.shape[0]
                
                output = output.mean(dim=0)
                test_loss_single += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_single += pred.eq(target.view_as(pred)).sum().item()
                n_single += target.shape[0]

        test_loss_single /= float(n_single)
        print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation' if is_test_set else 'Evaluation',
            test_loss_single, correct_single, n_single, 100. * correct_single / float(n_single)))
        
        for head in correct:
            correct[head] /= float(n[head])
            
        return correct, correct_single/float(n_single)
    
    else:
        # Single model with potential ensemble heads
        model = model_or_models
        model.eval()
        
        test_loss = defaultdict(lambda: 0)
        correct = defaultdict(lambda: 0)
        n = defaultdict(lambda: 0)
        test_loss_single = 0
        correct_single = 0
        n_single = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if args.fp16: data = data.half()
                if hasattr(model, 't'):
                    model.t = target
                output = model(data)
                
                if args.num_ensemble > 1:
                    output = torch.transpose(output, 0, 1)  # output size = [#ensemble, batch, #class]
                    
                    for head in range(len(output)):
                        test_loss[head] += F.cross_entropy(output[head], target, reduction='sum').item()
                        pred = output[head].argmax(dim=1, keepdim=True)
                        correct[head] += pred.eq(target.view_as(pred)).sum().item()
                        n[head] += target.shape[0]
                    
                    test_loss[head] /= float(n[head])
                    output = output.mean(dim=0)
                
                test_loss_single += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct_single += pred.eq(target.view_as(pred)).sum().item()
                n_single += target.shape[0]
                
        test_loss_single /= float(n_single)
        print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation' if is_test_set else 'Evaluation',
            test_loss_single, correct_single, n_single, 100. * correct_single / float(n_single)))
        
        for head in correct:
            correct[head] /= float(n[head])
            
        return correct, correct_single/float(n_single)


def main():
    start_train_time = time.time()
    args = utils.parse_args()
    utils.print_args(args)
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
    print('='*80)
    torch.manual_seed(args.seed)

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
    elif args.data == 'cifar100':
        train_loader, test_loader = get_cifar100_dataloaders(args)

    # Determine if we're using ensemble list or single model with multiple heads
    use_ensemble_list = args.baseline_ensemble > 1

    if use_ensemble_list:
        # Create separate models for ensemble
        ensemble_models = []
        for i in range(args.baseline_ensemble):
            if args.model == 'ResNet50':
                model = ResNet50(args).to(device)
            elif args.model == 'ResNet18':
                model = ResNet18(args).to(device)
            elif args.model == 'ResNet34':
                model = ResNet34(args).to(device)
            elif args.model == 'WideResNet':
                model = WideResNet_28(args).to(device)
            else:
                raise Exception('Model not implemented for ensemble list.')

            ensemble_models.append(model)
            args.seed = args.seed + i
        model = ensemble_models
    else:
        # Create single model with potential ensemble heads
        if args.model == 'ResNet50':
            model = ResNet50(args).to(device)
        elif args.model == 'ResNet18':
            model = ResNet18(args).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(args).to(device)
        elif args.model == 'WideResNet':
            model = WideResNet_28(args).to(device)
        else:
            raise Exception('Model not implemented yet.')

    if use_ensemble_list:
        # Create optimizers for each model
        ensemble_optimizers = []
        for i in range(args.baseline_ensemble):
            if args.optimizer == 'sgd':
                optimizer = optim.SGD(ensemble_models[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)
            elif args.optimizer == 'adam':
                optimizer = optim.Adam(ensemble_models[i].parameters(), lr=args.lr, weight_decay=args.l2)
            else:
                raise Exception('Unknown optimizer: {0}'.format(args.optimizer))
            ensemble_optimizers.append(optimizer)
        optimizer = ensemble_optimizers
        
        # Create schedulers for each optimizer
        ensemble_scheduler = []
        for i in range(args.baseline_ensemble):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(ensemble_optimizers[i], 
                                                               milestones=[int(args.epochs * 0.25),int(args.epochs * 0.5), int(args.epochs * 0.75)],
                                                               last_epoch=-1)
            ensemble_scheduler.append(lr_scheduler)
        lr_scheduler = ensemble_scheduler
    else:
        # Create single optimizer for the model
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')
            
        # Create scheduler for the optimizer
        if args.lr_scheduler == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=0.00001, patience=10)
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(args.epochs * 0.25), int(args.epochs * 0.5), int(args.epochs * 0.75)],
                                                            last_epoch=-1)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            original_acc = evaluate(args, model, device, test_loader)

    if args.fp16:
        print('FP16')
        if not use_ensemble_list:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                    dynamic_loss_args={'init_scale': 2 ** 16})
            model = model.half()
        else:
            for i in range(len(ensemble_models)):
                ensemble_optimizers[i] = FP16_Optimizer(ensemble_optimizers[i], static_loss_scale=None, dynamic_loss_scale=True,
                                                     dynamic_loss_args={'init_scale': 2 ** 16})
                ensemble_models[i] = ensemble_models[i].half()

    # Initialize masking
    if use_ensemble_list:
        ensemble_masks = [None for i in range(args.baseline_ensemble)]
        if not args.density == 1:
            ensemble_masks = []
            for i in range(args.baseline_ensemble):
                decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs))
                mask = Masking(ensemble_optimizers[i], death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                               redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
                mask.add_module(ensemble_models[i], sparse_init=args.sparse_init, density=args.density,load_masks = None)
                ensemble_masks.append(mask)
        mask = ensemble_masks
    else:
        mask = None
        if not args.density == 1:
            decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs))
            mask = Masking(optimizer, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args, train_loader=train_loader)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density, load_masks=None)

    best_acc = 0.0

    # Setup save path
    if use_ensemble_list:
        save_path = 'baseline_ensemble/' + str(args.model)
        save_subfolder = os.path.join(save_path, str(args.model) + '_base_ens=' + str(args.baseline_ensemble) + 
                                    '_density' + str(args.density) + '_seed' + str(args.seed) + 
                                    '_blocks_in_head' + str(args.blocks_in_head))
    else:
        save_path = 'save_weights/' + str(args.model)
        save_subfolder = os.path.join(save_path,
                                    str(args.model) + '_base_ens=' + str(args.baseline_ensemble) + '_density' + str(
                                        args.density) + 'num_ensemble' + str(args.num_ensemble) + '_seed' + str(
                                        args.seed) + '_blocks_in_head' + str(args.blocks_in_head) + 
                                    'update_frequency' + str(args.update_frequency) + 'epochs' + str(args.epochs))
    
    if not os.path.exists(save_subfolder): 
        os.makedirs(save_subfolder, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train(args, model, device, train_loader, optimizer, epoch, mask)
        
        if use_ensemble_list:
            for i in range(args.baseline_ensemble):
                lr_scheduler[i].step()
        else:
            if args.lr_scheduler == 'plateau':
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()

        val_acc_heads, val_acc = evaluate(args, model, device, test_loader)
        
        wandb_dict = {"val_acc": val_acc, "epoch": epoch}
        if use_ensemble_list:
            if args.baseline_ensemble > 1:
                for head in val_acc_heads:
                    wandb_dict[f"val_acc_head{head}"] = val_acc_heads[head]
        else:
            if args.num_ensemble > 1:
                for head in val_acc_heads:
                    wandb_dict[f"acc_per_head/val_acc_head{head}"] = val_acc_heads[head]
        
        wandb.log(wandb_dict)

        if val_acc > best_acc:
            print('Saving model')
            best_acc = val_acc
            if use_ensemble_list:
                for index, model_instance in enumerate(ensemble_models):
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model_instance.state_dict(),
                        'optimizer': ensemble_optimizers[index].state_dict(),
                    }, filename=os.path.join(save_subfolder, f'model_final_{index}.pth'))
            else:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_subfolder, 'model_final.pth'))

        if use_ensemble_list:
            print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                ensemble_optimizers[0].param_groups[0]['lr'], time.time() - t0))
        else:
            print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))

    print('Best accuracy is:', best_acc)
    total_training_hours = (time.time() - start_train_time) / 3600.
    print('Total training time (hours) is:', total_training_hours)
    wandb.log({"best_acc": best_acc, "total_training_hours": total_training_hours})


if __name__ == '__main__':
    main()
