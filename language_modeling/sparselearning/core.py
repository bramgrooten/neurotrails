from __future__ import print_function
import torch
import copy
import numpy as np
import math
import wandb
from sparselearning.decay import CosineDecay, LinearDecay, ConstantDecay


class Masking(object):
    """
    Controls the dynamic sparsity patterns in neural networks during training.
    
    This class manages the complete lifecycle of sparse training, including initialization,
    weight pruning, weight regrowth across the network.
    It supports various sparse training algorithms through different combinations of 
    prune_mode and growth_mode parameters. For example:
    - RigL: prune_mode='magnitude', growth_mode='gradient'
    - SET: prune_mode='magnitude', growth_mode='random'
    
    """
    def __init__(
            self,
            optimizer,
            growth_prune_ratio=1.0,
            redistribution_mode='none',
            threshold=0.001,
            args=None,
            distributed=False,
            device=None,
    ):
        self.args = args
        self.optimizer = optimizer
        self.distributed = distributed
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = device

        self.growth_mode = args.growth
        self.prune_mode = args.prune
        self.growth_prune_ratio = growth_prune_ratio
        self.redistribution_mode = redistribution_mode

        self.prune_funcs = {}
        self.prune_funcs['magnitude'] = self.magnitude_prune
        self.prune_funcs['SET'] = self.magnitude_and_negativity_prune
        self.prune_funcs['threshold'] = self.threshold_prune

        self.growth_funcs = {}
        self.growth_funcs['random'] = self.random_growth
        self.growth_funcs['momentum'] = self.momentum_growth
        self.growth_funcs['momentum_neuron'] = self.momentum_neuron_growth

        self.masks = {}
        self.final_masks = {}
        self.grads = {}
        self.scores = {}
        self.modules = []
        self.names = []

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
        self.prune_rate = args.prune_rate
        self.name2prune_rate = {}
        self.name2density = {}
        self.steps = 0

        # global growth/prune state
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = args.update_frequency

        self.set_prune_rate_decay()
        self.set_density_decay()
        self.set_temperature_decay()

    def synchronize_masks(self):
        """ Synchronize masks across GPUs. """
        if self.distributed:
            for name in self.masks.keys():
                torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

    def init_sparse_masks(self, erk_power_scale=1.0):
        if self.args.density_decay == 'constant':
            density = self.args.density
        else:
            density = self.args.initial_density

        if self.sparse_init == 'uniform':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                    self.baseline_nonzero += weight.numel()*density
            self.apply_mask()

        elif self.sparse_init == 'Multi_Output':
            print('initialize by Multi_Output')
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

            print(f"Overall density {total_nonzero / total_params}")

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
            print(f"Overall density {total_nonzero / total_params}")
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
            print(f"Overall density {total_nonzero / total_params}")


        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for over-paremeters

        self.init_prune_rate(self.prune_rate)
        self.init_density_per_layer()
        self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total initial parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))


    def init_prune_rate(self, prune_rate):
        for name in self.masks:
            self.name2prune_rate[name] = prune_rate

    def init_density_per_layer(self):
        """ Get the initialized density of each layer. """
        for name in self.masks:
            mask = self.masks[name]
            num_nonzero = mask.float().sum().item()
            total_weights = mask.numel()
            density = num_nonzero / total_weights
            self.name2density[name] = density

    def step(self):
        """
        Executes a single optimization step in the sparse training loop.
    
        This function:
        1. Performs the standard optimizer step to update weights
        2. Applies sparsity masks to maintain network sparsity
        3. Updates prune rates according to the configured decay schedule
        4. Periodically prunes weights and logs sparsity statistics
        
        The prune rate controls the proportion of weights pruned during topology
        evolution, and can follow either a cosine or constant decay schedule.
        """
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.density_decay.step()
        self.temperature_decay.step()

        for name in self.masks:
            if self.args.prune_rate_decay == 'cosine':
                self.name2prune_rate[name] = self.prune_rate_decay.get_current_value()
            elif self.args.prune_rate_decay == 'constant':
                self.name2prune_rate[name] = self.args.prune_rate
            self.prune_rate = self.name2prune_rate[name]

        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.steps % self.prune_every_k_steps == 0:
                self.set_new_layer_densities()
                self.truncate_weights()
                self.print_nonzero_counts()


    def add_module(self, module, density, sparse_init='ER'):
        self.sparse_init = sparse_init
        self.modules.append(module)
        print('adding module')
        for name, tensor in module.named_parameters():
            print(f'(len: {len(tensor.size())}) size of {name}: {tensor.size()}')

            if self.args.dense_embedding and 'embed' in name:
                print(f'Keeping embedding layer dense: {name}')
                continue  # skip embedding layer, if requested

            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                # self.masks[name] = torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()  # old version
                self.masks[name] = torch.ones_like(tensor, dtype=tensor.dtype, requires_grad=False).cuda()

        # self.remove_weight_partial_name('bias')
        # self.remove_type(torch.nn.BatchNorm2d)
        # self.remove_type(torch.nn.BatchNorm1d)
        self.init_sparse_masks()

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
        self.synchronize_masks()
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

                    # not used for Adam: self.optimizer.state[tensor].keys() has 'step', 'exp_avg', 'exp_avg_sq'
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights(self):
        """
        Core function responsible for dynamic neural network topology evolution through structured sparsity.
        
        This method implements the sparse training paradigm by first pruning (truncating) weights based on
        specified criteria, then activating new parameters to maintain a constant sparsity level. The process
        follows these steps:
        
        1. Collect network statistics for informed decision-making
        2. Calculate parameter redistribution across layers
        3. Remove parameters based on the specified prune_mode
        4. Regrow parameters using the specified growth_mode
        
        Common weight pruning strategies include:
        - magnitude: Remove smallest magnitude weights (most common)
        - soft_magnitude: Rank smallest magnitude weights and remove based on probability
        - SET: Remove smallest and most negative weights
        - global_magnitude: Apply magnitude pruning globally across all layers
        
        Common weight regrowth strategies include:
        - random: Randomly activate new parameters (used in SET method)
        - gradient: Use gradient information to guide parameter activation (used in RigL method)
        - momentum: Leverage momentum data for intelligent regrowth
        """
        self.gather_statistics()
        name2regrowth = self.calc_growth_redistribution()

        total_removed = 0
        if self.prune_mode == 'global_magnitude':
            total_removed = self.global_magnitude_prune()
        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # prune
                    if self.prune_mode == 'magnitude':
                        new_mask = self.magnitude_prune(mask, weight, name)
                    # elif self.prune_mode == 'mag_gra':
                    #     new_mask = self.mag_gra(mask, weight, name, epoch)
                    elif self.prune_mode == 'magnitude_soft':
                        new_mask = self.magnitude_soft_prune(weight, name)
                    elif self.prune_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_prune(mask, weight, name)
                    elif self.prune_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.prune_mode == 'threshold':
                        new_mask = self.threshold_prune(mask, weight, name)
                    elif self.prune_mode == 'magnitude_increase':
                        new_mask = self.magnitude_increase(weight, mask, name)

                    total_removed += self.name2nonzeros[name] - new_mask.float().sum().item()
                    self.masks[name] = new_mask.to(weight.dtype)

        # Do we want to re-init weight values here (between pruning and growing)?
        # If newly grown weights should just start from value 0, then just apply_mask
        # If 'no', then no applying mask between pruning and growing (immediately regrown weights retain their values)
        if self.args.reinit == 'zero':
            self.apply_mask()
        elif self.args.reinit == 'original':
            self.reinit_weights_original_distribution()

        # growing
        if self.growth_mode == 'global_momentum':
            _ = self.global_momentum_growth(total_removed + self.adjusted_growth)
        else:
            if self.prune_mode == 'threshold':
                expected_killed = sum(name2regrowth.values())
                if total_removed < (1.0-self.tolerance)*expected_killed:
                    self.threshold *= 2.0
                elif total_removed > (1.0+self.tolerance) * expected_killed:
                    self.threshold *= 0.5

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()

                    if self.prune_mode == 'threshold':
                        total_regrowth = math.floor((total_removed/float(expected_killed))*name2regrowth[name]*self.growth_prune_ratio)
                    elif self.redistribution_mode == 'none':

                        # if name not in self.name2baseline_nonzero:
                        #     self.name2baseline_nonzero[name] = self.name2nonzeros[name]
                        # old = self.name2baseline_nonzero[name]
                        # new = new_mask.sum().item()
                        # total_regrowth = int(old-new)

                        left_after_prune = new_mask.float().sum().item()
                        desired_num = math.ceil(self.name2density[name] * self.masks[name].numel())
                        total_regrowth = int(desired_num - left_after_prune)

                        # print(f'total_regrowth: {total_regrowth}  name: {name}, left_after_prune: {left_after_prune}, desired_num: {desired_num}')

                        assert total_regrowth >= 0, "total_regrowth should be >= 0"

                    elif self.prune_mode == 'global_magnitude':
                        expected_removed = self.baseline_nonzero*self.name2prune_rate[name]
                        expected_vs_actual = total_removed/expected_removed
                        total_regrowth = math.floor(expected_vs_actual*name2regrowth[name]*self.growth_prune_ratio)
                    else:
                        total_regrowth = math.floor(name2regrowth[name]*self.growth_prune_ratio)

                    # growth
                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'gradient':  # RigL
                        new_mask, grad = self.gradient_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'momentum_neuron':
                        new_mask = self.momentum_neuron_growth(name, new_mask, total_regrowth, weight)

                    elif self.growth_mode == 'mix_growth':
                        new_mask = self.mix_growth(name, new_mask, total_regrowth, weight)

                    else:
                        raise ValueError(f"Unknown growth mode: {self.growth_mode}")

                    self.masks[name] = new_mask.to(weight.dtype)

        self.apply_mask()


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
                self.name2nonzeros[name] = mask.float().sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                prune_rate = self.name2prune_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        prune_rate = min(sparsity, prune_rate)
                num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
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
                #prune_rate = min(self.name2prune_rate[name], max(0.05, (self.name2zeros[name]/float(self.masks[name].numel()))))
                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                prune_rate = self.name2prune_rate[name]
                if sparsity < 0.2:
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        prune_rate = min(sparsity, prune_rate)
                num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
                #num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])
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
                    prune
    '''
    def magnitude_increase(self, weight, mask, name): # lsw addition
        prune_rate = self.name2prune_rate[name]
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        pruning_number = self.name2nonzeros[name] * prune_rate
        k = math.ceil(self.name2zeros[name] + pruning_number)
        threshold = x[k - 1].item()
        # magIN_num = (torch.abs(weight) > torch.abs(self.pre_tensor[name])).sum().item()
        # smaller_num = (torch.abs(weight) < torch.abs(self.pre_tensor[name])).sum().item()
        # bigThan_mean = (torch.abs(weight) > threshold).sum().item()
        # print('mag increase number', magIN_num/num_nonzero, 'threshold', bigThan_mean/num_nonzero)
        return (torch.abs(weight) > torch.abs(self.pre_tensor[name])) | (torch.abs(weight) > threshold)  # check if mask if right?

    def threshold_prune(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2prune_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def mag_gra(self, mask, weight, name, epoch):
        if epoch <= self.args.fminestone:
            lamda = 0.999
        elif epoch <= self.args.sminestone:
            lamda = 0.995
        else:
            lamda = 0.99
        grad = weight.grad.clone()
        score = lamda*torch.abs(weight) + (1-lamda)*torch.abs(grad)
        num_remove = math.ceil(self.name2prune_rate[name] * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        # num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])

        num_zeros = self.name2zeros[name]
        x, idx = torch.sort(torch.abs(score.data.view(-1)))
        k = math.ceil(num_zeros + num_remove)
        mask.data.view(-1)[idx[:k]] = 0.0
        return mask

    def kernel_pruning(self, mask, weight, name):

        score = torch.clone(weight.grad * weight).detach().abs_()

        num_remove = math.ceil(self.name2prune_rate[name] * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])

        num_zeros = self.name2zeros[name]
        x, idx = torch.sort(score.data.view(-1))
        k = math.ceil(num_zeros + num_remove)
        mask.data.view(-1)[idx[:k]] = 0.0
        return mask

    def magnitude_prune(self, mask, weight, name):
        sparsity = self.name2zeros[name]/float(self.masks[name].numel())
        prune_rate = self.name2prune_rate[name]
        if sparsity < 0.2:
            expected_variance = 1.0/len(list(self.name2variance.keys()))
            actual_variance = self.name2variance[name]
            expected_vs_actual = expected_variance/actual_variance
            if expected_vs_actual < 1.0:
                prune_rate = min(sparsity, prune_rate)
                print(name, expected_variance, actual_variance, expected_vs_actual, prune_rate)
        num_remove = math.ceil(prune_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]
        num_nonzero = n-num_zeros

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def magnitude_soft_prune(self, weight, name):
        """
        Soft magnitude pruning with temperature-scaled sampling.
        from Zhang et al. https://arxiv.org/abs/2501.19107

        To avoid errors:
        If the probability vector is too sparse to draw `num_to_stay`
        unique indices, the current mask is returned unchanged.
        """
        # take the absolute value of the masked weights
        matrix = torch.abs(weight * self.masks[name])

        num_active = self.masks[name].float().sum().item()
        num_to_stay = math.ceil(num_active * (1 - self.name2prune_rate[name]))

        flat_matrix = matrix.flatten()
        flat_matrix = torch.where(torch.isnan(flat_matrix), torch.zeros_like(flat_matrix), flat_matrix)
        flat_matrix = torch.where(torch.isinf(flat_matrix), torch.zeros_like(flat_matrix), flat_matrix)

        T = self.temperature_decay.get_current_value()
        flat_matrix = flat_matrix.float() ** T

        # define probabilities of weights to stay unpruned
        total = flat_matrix.sum()
        if total == 0:
            return self.masks[name].clone()
        probs = flat_matrix / total

        if probs.numel() > 2 ** 24:  # avoid CUDA limit of torch.multinomial
            # numpy handles > 2**24 (~16M) categories fine
            probs = probs.detach().cpu().numpy()

            if np.flatnonzero(probs).size < num_to_stay:
                # if not enough non-zero probs, return the original mask
                return self.masks[name].clone()

            keep_idx_np = np.random.choice(probs.size, size=num_to_stay, replace=False, p=probs)
            keep_idx = torch.from_numpy(keep_idx_np).to(weight.device, dtype=torch.long)
        else:
            if torch.nonzero(probs).squeeze().numel() < num_to_stay:
                # if not enough non-zero probs, return the original mask
                return self.masks[name].clone()

            keep_idx = torch.multinomial(probs, num_to_stay, replacement=False)

        new_mask = torch.zeros_like(weight, device=self.device)
        new_mask.view(-1)[keep_idx] = 1
        return new_mask

    def global_magnitude_prune(self):
        prune_rate = 0.0
        for name in self.name2prune_rate:
            if name in self.masks:
                prune_rate = self.name2prune_rate[name]
        tokill = math.ceil(prune_rate*self.baseline_nonzero)
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
                total_new_nonzeros += new_mask.float().sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_prune(self, mask, weight, name):
        num_remove = math.ceil(self.name2prune_rate[name]*self.name2nonzeros[name])
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
        """
        This function implements the random growth strategy for sparse neural networks,
        which is used in algorithms like SET. It randomly
        selects zero-valued positions in the weight matrix to be activated, with the
        total number of new connections controlled by the total_regrowth parameter.
        """
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
        """
        This function implements the gradient-based growth strategy for sparse neural networks,
        which is a key component of methods like RigL (Rigged Lottery Tickets). It prioritizes
        regrowth at zero-valued positions where gradients have the highest magnitude, indicating
        where new connections would have the most immediate impact on loss reduction.
        """
        grad = weight.grad.clone()
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask, grad

    def mix_growth(self, name, new_mask, total_regrowth, weight):
        gradient_grow = int(total_regrowth * self.args.mix)
        random_grow = total_regrowth - gradient_grow
        grad = weight.grad.clone()
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

    def print_nonzero_counts(self):
        total_active = 0
        total_params = 0
        total_active_incl_bias = 0
        total_params_incl_bias = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                total_params_incl_bias += tensor.numel()
                if name not in self.masks:
                    total_active_incl_bias += tensor.numel()
                    continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                total_active += num_nonzeros
                total_active_incl_bias += num_nonzeros
                total_params += mask.numel()
                if name in self.name2variance:
                    val = '{0}: {1}->{2}, density: {3:.3f}, proportion: {4:.4f}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), self.name2variance[name])
                    print(val)
                else:
                    print(name, num_nonzeros)

        if self.args.single_gpu:
            wandb.log({
                "sparsity/density": total_active / total_params,
                "sparsity/density_incl_bias": total_active_incl_bias / total_params_incl_bias,
                "sparsity/total_active": total_active,
                "sparsity/total_active_incl_bias": total_active_incl_bias,
                "sparsity/total_params": total_params,
                "sparsity/total_params_incl_bias": total_params_incl_bias,
                "sparsity/density_decay": self.density_decay.get_current_value(),
                "sparsity/prune_rate": self.prune_rate_decay.get_current_value(),
                "sparsity/temperature": self.temperature_decay.get_current_value(),
            })

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print(f'prune rate {name}: {self.name2prune_rate[name]}')
                break  # only print the first tensor, prune_rate is same across layers in our experiments

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

    def set_prune_rate_decay(self):
        if self.args.prune_rate_decay == 'cosine':
            self.prune_rate_decay = CosineDecay(
                init_value=self.args.prune_rate,
                T_max=self.args.num_training_steps,
                eta_min=0.005,
            )
        elif self.args.prune_rate_decay == 'linear':
            self.prune_rate_decay = LinearDecay(
                init_value=self.args.prune_rate,
                final_value=0.005,
                num_steps=self.args.num_training_steps,
            )
        elif self.args.prune_rate_decay == 'constant':
            self.prune_rate_decay = ConstantDecay(self.args.prune_rate)
        else:
            raise Exception(f'Unknown prune_rate_decay mode: {self.args.prune_rate_decay}')

    def set_density_decay(self):
        if self.args.density_decay == 'cosine':
            self.density_decay = CosineDecay(
                init_value=self.args.initial_density,
                T_max=self.args.num_training_steps,
                eta_min=self.args.density,
            )
        elif self.args.density_decay == 'linear':
            self.density_decay = LinearDecay(
                init_value=self.args.initial_density,
                final_value=self.args.density,
                num_steps=self.args.num_training_steps,
            )
        elif self.args.density_decay == 'constant':
            self.density_decay = ConstantDecay(self.args.density)
        else:
            raise Exception(f'Unknown density_decay mode: {self.args.density_decay}')

    def set_temperature_decay(self):
        if self.args.temperature_decay == 'linear':
            self.temperature_decay = LinearDecay(
                init_value=self.args.init_temperature,
                final_value=self.args.temperature,
                num_steps=self.args.num_training_steps,
            )
        elif self.args.temperature_decay == 'constant':
            self.temperature_decay = ConstantDecay(self.args.temperature)
        else:
            raise Exception(f'Unknown temperature_decay mode: {self.args.temperature_decay}')

    def set_new_layer_densities(self):
        if self.args.density_decay != 'constant':
            total = 0
            total_active = 0
            for name in self.masks:
                total_active += self.masks[name].float().sum().item()
                total += self.masks[name].numel()

            prev_density = total_active / total
            new_density = self.density_decay.get_current_value()
            cur_dens_decay_factor = new_density / prev_density
            print(f'cur_dens_decay_factor: {cur_dens_decay_factor}  prev_density: {prev_density}  new_density: {new_density}  total: {total}  total_active: {total_active}')

            if self.args.single_gpu:
                wandb.log({
                    "sparsity/density_decay_factor": cur_dens_decay_factor,
                })

            for name in self.masks:
                self.name2density[name] = self.name2density[name] * cur_dens_decay_factor
                self.name2density[name] = min(self.name2density[name], 1)
                # assert 0 <= self.name2density[name] <= 1, \
                #     f'Density {self.name2density[name]} of layer {name} out of range [0, 1]'

    def reinit_weights_original_distribution(self):
        """Reinitialize pruned weights using the original initialization scheme."""
        for module in self.modules:
            for name, param in module.named_parameters():
                if name not in self.masks:
                    continue
                inactive_mask = (self.masks[name] == 0)

                embed = False
                if 'embed' in name:
                    embed = True

                with torch.no_grad():
                    temp_layer = torch.nn.Parameter(torch.empty_like(param))
                    weight_init(temp_layer, embedding=embed)
                    param.data[inactive_mask] = temp_layer.data[inactive_mask]
        # all weights have non-zero values now,
        # but this will be solved when we .apply_mask after growing new connections

def weight_init(weight, embedding=False):
    """Initialize weights using the original initialization scheme."""
    if embedding:
        std_embedding = (2 / 5) ** 0.5  # approx 0.632
        weight.data.normal_(mean=0.0, std=std_embedding)
    else:
        # std = config.initializer_range
        std = 0.02  # default value
        weight.data.normal_(mean=0.0, std=std)
