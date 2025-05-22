from __future__ import print_function
import torch
import torch.optim as optim
import copy
import numpy as np
import math


class CosineDecay(object):
    """
    Implements cosine annealing schedule for pruning rate decay in sparse training.
    
    During Dynamic Sparse Training (DST), this class manages the gradual reduction
    of the pruning rate (death rate) over the training process. The death rate
    typically starts high (e.g., 50% of available parameters being pruned/regrown 
    per iteration) and smoothly decays to a minimum value by the end of training
    following a cosine curve.
    """
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum',
                 redistribution_mode='momentum', threshold=0.001, args=None, train_loader=None, distributed = False):
        '''
        Controls the dynamic sparsity patterns in neural networks during training.
        
        This class manages the complete lifecycle of sparse training, including initialization,
        weight pruning (death), weight regrowth across the network.
        It supports various sparse training algorithms through different combinations of 
        death_mode and growth_mode parameters. For example:
        - RigL: death_mode='magnitude', growth_mode='gradient'
        - SET: death_mode='magnitude', growth_mode='random'
        '''
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader

        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.distributed = distributed

        self.death_funcs = {}
        self.death_funcs['magnitude'] = self.magnitude_death
        self.death_funcs['SET'] = self.magnitude_and_negativity_death
        self.death_funcs['threshold'] = self.threshold_death

        self.growth_funcs = {}
        self.growth_funcs['random'] = self.random_growth
        self.growth_funcs['momentum'] = self.momentum_growth
        self.growth_funcs['momentum_neuron'] = self.momentum_neuron_growth

        self.masks = {}
        self.final_masks = {}
        self.grads = {}
        self.nonzero_masks = {}
        self.scores = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0

        # global growth/death state
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency

    def synchronism_masks(self):
        '''
        For mask synchronization between GPUs during distributed training. 
        '''
        if self.distributed:
            for name in self.masks.keys():
                torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, load_masks=None):
        '''
        Initialize sparse neural network topology with controllable sparsity patterns.
    
        This function establishes the initial sparse topology required for sparse-to-sparse
        training methods like DST (Dynamic Sparse Training). It supports multiple initialization
        strategies:
    
        - Loading pre-existing masks from checkpoints
        - Uniform random initialization with fixed density
        - Global magnitude-based pruning
        - ERK (Erdős-Rényi-Kernel) method with configurable scaling
        '''
        self.density = density
        
        # If loading from checkpoint, directly apply the saved masks.
        if load_masks is not None:
            print('Loading pre-saved masks from the checkpoint...')
            self.masks = load_masks
            
            print('loading masks completed')
            self.apply_mask()
        elif self.sparse_init == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw

                    # self.masks[name][:] = (torch.rand(weight.shape) < density).float().data #lsw
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()
        elif self.sparse_init == 'pruning':
            print('initialize by pruning')
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - self.PF_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

        elif self.sparse_init == 'Multi_Output':
            print('initialize by Extreme Output')
            total_params = 0
            self.baseline_nonzero = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
                self.baseline_nonzero += weight.numel()*density
            
            remain_density = float(self.baseline_nonzero/total_params)
            print('current density is:', remain_density)
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - remain_density)
                    n_ones = n_param * remain_density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()

            print(f"Overall sparsity {total_nonzero / total_params}")

        elif self.sparse_init == 'fixed_ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        elif self.sparse_init == 'fixed_ER':
            print('initialize by fixed_ER')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density
                    n_in, n_out = mask.size()[:2]
                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                (n_in+n_out) / (n_in * n_out)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks)

        self.init_death_rate(self.death_rate)
        self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

    def init_death_rate(self, death_rate):
        for name in self.masks:
            self.name2death_rate[name] = death_rate

    def step(self):
        '''
        Executes a single optimization step in the sparse training loop.
    
        This function:
        1. Performs the standard optimizer step to update weights
        2. Applies sparsity masks to maintain network sparsity
        3. Updates death rates according to the configured decay schedule
        4. Periodically prunes weights and logs sparsity statistics
        
        The death rate controls the proportion of weights pruned during topology
        evolution, and can follow either a cosine or constant decay schedule.
        '''
        self.optimizer.step()
        self.apply_mask()
        self.death_rate_decay.step()
        for name in self.masks:
            if self.args.decay_schedule == 'cosine':
                self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])
            elif self.args.decay_schedule == 'constant':
                self.name2death_rate[name] = self.args.death_rate
            self.death_rate = self.name2death_rate[name]
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.truncate_weights()
                self.print_nonzero_counts()

    def add_module(self, module, density, load_masks,sparse_init='ER'):
        self.sparse_init = sparse_init
        self.modules.append(module)
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                # self.final_masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        # print('Removing biases...')
        # self.remove_weight_partial_name('bias')
        # print('Removing 2D batch norms...')
        # self.remove_type(nn.BatchNorm2d)
        # print('Removing 1D batch norms...')
        # self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=self.args.density, load_masks = load_masks)
        # self.ini_layer_pruning_rate()

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        self.synchronism_masks()
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]


    def truncate_weights(self, step=None):
        '''
        Core function responsible for dynamic neural network topology evolution through structured sparsity.
        
        This method implements the sparse training paradigm by first pruning (truncating) weights based on
        specified criteria, then activating new parameters to maintain a constant sparsity level. The process
        follows these steps:
        
        1. Collect network statistics for informed decision-making
        2. Calculate parameter redistribution across layers
        3. Remove parameters based on the specified death_mode
        4. Regrow parameters using the specified growth_mode
        
        Common weight pruning strategies include:
        - magnitude: Remove smallest magnitude weights (most common)
        - SET: Remove smallest and most negative weights
        - global_magnitude: Apply magnitude pruning globally across all layers
        
        Common weight regrowth strategies include:
        - random: Randomly activate new parameters (used in SET method)
        - gradient: Use gradient information to guide parameter activation (used in RigL method)
        - momentum: Leverage momentum data for intelligent regrowth
        '''        
        self.gather_statistics()
        name2regrowth = self.calc_growth_redistribution()

        total_nonzero_new = 0
        total_removed = 0
        if self.death_mode == 'global_magnitude':
            total_removed = self.global_magnitude_death()
        else:
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.death_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)
                    elif self.death_mode == 'magnitude_increase':
                        new_mask = self.magnitude_increase(weight, mask, name)

                    total_removed += self.name2nonzeros[name] - new_mask.sum().item()
                    self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())
                    self.masks[name][:] = new_mask
                    self.nonzero_masks[name] = new_mask.float()

        # self.apply_mask()
        if self.growth_mode == 'global_momentum':
            total_nonzero_new = self.global_momentum_growth(total_removed + self.adjusted_growth)
        else:
            if self.death_mode == 'threshold':
                expected_killed = sum(name2regrowth.values())
                #print(expected_killed, total_removed, self.threshold)
                if total_removed < (1.0-self.tolerance)*expected_killed:
                    self.threshold *= 2.0
                elif total_removed > (1.0+self.tolerance) * expected_killed:
                    self.threshold *= 0.5

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    if self.death_mode == 'threshold':
                        total_regrowth = math.floor((total_removed/float(expected_killed))*name2regrowth[name]*self.growth_death_ratio)
                    elif self.redistribution_mode == 'none':
                        if name not in self.name2baseline_nonzero:
                            self.name2baseline_nonzero[name] = self.name2nonzeros[name]
                        old = self.name2baseline_nonzero[name]
                        new = new_mask.sum().item()
                        #print(old, new)
                        total_regrowth = int(old-new)
                    elif self.death_mode == 'global_magnitude':
                        expected_removed = self.baseline_nonzero*self.name2death_rate[name]
                        expected_vs_actual = total_removed/expected_removed
                        total_regrowth = math.floor(expected_vs_actual*name2regrowth[name]*self.growth_death_ratio)
                    else:
                        total_regrowth = math.floor(name2regrowth[name]*self.growth_death_ratio)

                    # growth
                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'gradient':
                        # implementation for Rigging Ticket
                        new_mask, grad = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                    elif self.growth_mode == 'momentum_neuron':
                        new_mask = self.momentum_neuron_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'mix_growth':
                        new_mask = self.mix_growth(name, new_mask, total_regrowth, weight)

                    new_nonzero = new_mask.sum().item()

                    # exchanging masks
                    self.masks.pop(name)
                    self.masks[name] = new_mask.float()
                    total_nonzero_new += new_nonzero
        self.apply_mask()

        # Some growth techniques and redistribution are probablistic and we might not grow enough weights or too much weights
        # Here we run an exponential smoothing over (death-growth) residuals to adjust future growth
        # self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        # self.adjusted_growth = 0.25*self.adjusted_growth + (0.75*(self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)
        # print(self.total_nonzero, self.baseline_nonzero, self.adjusted_growth)

        if self.total_nonzero > 0:
            print('old, new nonzero count:', self.total_nonzero, total_nonzero_new, self.adjusted_growth)

    '''
                    REDISTRIBUTION
    '''
    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}

        self.total_variance = 0.0
        self.total_removed = 0
        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                if self.redistribution_mode == 'momentum':
                    grad = self.get_momentum_for_weight(tensor)
                    self.name2variance[name] = torch.abs(grad[mask.byte()]).mean().item()#/(V1val*V2val)
                elif self.redistribution_mode == 'magnitude':
                    self.name2variance[name] = torch.abs(tensor)[mask.byte()].mean().item()
                elif self.redistribution_mode == 'nonzeros':
                    self.name2variance[name] = float((torch.abs(tensor) > self.threshold).sum().item())
                elif self.redistribution_mode == 'none':
                    self.name2variance[name] = 1.0
                elif self.redistribution_mode == 'magnitude_increase':
                    # only calculate the increased weights
                    mask_increased = torch.abs(tensor) > torch.abs(self.pre_tensor[name])
                    # weights_increased = (torch.abs(tensor) - torch.abs(self.pre_tensor[name])).mean().item()
                    # print(name, "Weight increased:", weights_increased)
                    # include all the non-zero weights
                    self.name2variance[name] = (torch.abs(tensor[mask_increased.byte()]) - torch.abs(self.pre_tensor[name][mask_increased.byte()])).mean().item()
                    # self.name2variance[name] = torch.abs(tensor[mask.byte()] - self.pre_tensor[name][mask.byte()]).mean().item()
                    # print("name", name, "abs_MI",self.name2variance[name])# mean of ABS of magnitude increased weights
                    # print("abs_M",torch.abs(tensor[mask.byte()] - self.pre_tensor[name][mask.byte()]).mean().item())  # mean() of absolute of all weights magnitude increased
                elif self.redistribution_mode == 'uniform_distribution':
                    self.name2variance[name] = 1
                else:
                    print('Unknown redistribution mode:{0}'.format(self.redistribution_mode))
                    raise Exception('Unknown redistribution mode!')

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                self.total_removed += num_remove
                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

    def calc_growth_redistribution(self):
        num_overgrowth = 0
        total_overgrowth = 0
        residual = 0
        for name in self.name2variance:
            self.name2variance[name] /= self.total_variance

        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        expected_var = 1.0/len(self.name2variance)
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                #death_rate = min(self.name2death_rate[name], max(0.05, (self.name2zeros[name]/float(self.masks[name].numel()))))
                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                death_rate = self.name2death_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        death_rate = min(sparsity, death_rate)
                num_remove = math.ceil(death_rate*self.name2nonzeros[name])
                #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
                num_nonzero = self.name2nonzeros[name]
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name]*(self.total_removed+self.adjusted_growth))
                regrowth += mean_residual

                #if regrowth > max_regrowth:
                #    name2regrowth[name] = max_regrowth
                if regrowth > 0.99*max_regrowth:
                    name2regrowth[name] = 0.99*max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0: mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        return name2regrowth

    '''
                    DEATH
    '''
    def magnitude_increase(self, weight, mask, name): # lsw addition
        death_rate = self.name2death_rate[name]
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        pruning_number = self.name2nonzeros[name] * death_rate
        k = math.ceil(self.name2zeros[name] + pruning_number)
        threshold = x[k - 1].item()
        # magIN_num = (torch.abs(weight) > torch.abs(self.pre_tensor[name])).sum().item()
        # smaller_num = (torch.abs(weight) < torch.abs(self.pre_tensor[name])).sum().item()
        # bigThan_mean = (torch.abs(weight) > threshold).sum().item()
        # print('mag increase number', magIN_num/num_nonzero, 'threshold', bigThan_mean/num_nonzero)
        return (torch.abs(weight) > torch.abs(self.pre_tensor[name])) | (torch.abs(weight) > threshold)  # check if mask if right?

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def kernel_pruning(self, mask, weight, name):
        score = torch.clone(weight.grad * weight).detach().abs_()
        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        x, idx = torch.sort(score.data.view(-1))
        k = math.ceil(num_zeros + num_remove)
        mask.data.view(-1)[idx[:k]] = 0.0
        return mask

    def magnitude_death(self, mask, weight, name):
        sparsity = self.name2zeros[name]/float(self.masks[name].numel())
        death_rate = self.name2death_rate[name]
        if sparsity < 0.2:
            expected_variance = 1.0/len(list(self.name2variance.keys()))
            actual_variance = self.name2variance[name]
            expected_vs_actual = expected_variance/actual_variance
            if expected_vs_actual < 1.0:
                death_rate = min(sparsity, death_rate)
                print(name, expected_variance, actual_variance, expected_vs_actual, death_rate)
        num_remove = math.ceil(death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]
        num_nonzero = n-num_zeros
        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()
        return (torch.abs(weight.data) > threshold)

    def global_magnitude_death(self):
        death_rate = 0.0
        for name in self.name2death_rate:
            if name in self.masks:
                death_rate = self.name2death_rate[name]
        tokill = math.ceil(death_rate*self.baseline_nonzero)
        total_removed = 0
        prev_removed = 0
        while total_removed < tokill*(1.0-self.tolerance) or (total_removed > tokill*(1.0+self.tolerance)):
            total_removed = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    remain = (torch.abs(weight.data) > self.threshold).sum().item()
                    total_removed += self.name2nonzeros[name] - remain

            if prev_removed == total_removed: break
            prev_removed = total_removed
            if total_removed > tokill*(1.0+self.tolerance):
                self.threshold *= 1.0-self.increment
                self.increment *= 0.99
            elif total_removed < tokill*(1.0-self.tolerance):
                self.threshold *= 1.0+self.increment
                self.increment *= 0.99

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name][:] = torch.abs(weight.data) > self.threshold
        return int(total_removed)


    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown

            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()

        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)

        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''
    def random_growth(self, name, new_mask, total_regrowth, weight):
        '''
        This function implements the random growth strategy for sparse neural networks,
        which is used in algorithms like SET. It randomly
        selects zero-valued positions in the weight matrix to be activated, with the
        total number of new connections controlled by the total_regrowth parameter.
        '''
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability #lsw
        # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def kernel_gradient_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.grads[name]
        grad = grad * (new_mask == 0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        '''
        This function implements the gradient-based growth strategy for sparse neural networks,
        which is a key component of methods like RigL (Rigged Lottery Tickets). It prioritizes
        regrowth at zero-valued positions where gradients have the highest magnitude, indicating
        where new connections would have the most immediate impact on loss reduction.
        '''
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        return new_mask, grad

    def mix_growth(self, name, new_mask, total_regrowth, weight):
        gradient_grow = int(total_regrowth * self.args.mix)
        random_grow = total_regrowth - gradient_grow
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:gradient_grow]] = 1.0

        n = (new_mask == 0).sum().item()
        expeced_growth_probability = (random_grow / n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask = new_mask.bool() | new_weights

        return new_mask

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    print(val)
                else:
                    print(name, num_nonzeros)

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.name2death_rate[name]))
                break

    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        if self.args.reset_mom_zero:
                            print('zero')
                            self.optimizer.state[tensor][w][mask == 0] = 0
                        else:
                            print('mean')
                            self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights
