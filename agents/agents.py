import copy

import numpy as np
import random 

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from nets import MLP, ConvNet, DeepConvNet

from utils.continual_backprop.gnt import GnT, AdamGnT
from utils.continual_backprop.convGnT import ConvGnT


class BaseAgent:

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device):
        """
        Initializes the BaseAgent instance.

        Args:
            model_constructor (callable): A callable that constructs the model. The callable should return an instance of nn.Module.
            optimizer_cfg (dict): Configuration for the optimizer, including optimizer type and hyperparameters.
            loss_fn (callable): The loss function to be used for training. Should accept model predictions and target values as inputs.
            device (torch.device): The device on which the model and computations will run (e.g., 'cpu' or 'cuda').
        """

        super().__init__()

        self.model_constructor = model_constructor
        self.model = model_constructor()
        self.device = device
        self.model.to(device)
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = loss_fn
        self.get_optimizer()

    def get_optimizer(self):

        if self.optimizer_cfg.name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer_cfg.lr)
        elif self.optimizer_cfg.name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer_cfg.lr)
                      
    def compute_loss(self, x, y):

        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        return loss, logits

    def predict(self, x):
        return self.model(x)

    def step(self, x, y):

        loss, logits = self.compute_loss(x, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Metrics computed at every step.
        # Get the magnitude of the gradient
        grad_metrics = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'layer_norm' not in name and \
                'init_params' not in name and \
                    'original_last_layer_params' not in name:
                grad_metrics[f'agent/{name}-magnitude'] = torch.norm(param.grad)
                grad_metrics[f'agent/{name}-frac-zero'] = torch.mean((param.grad == 0).float()).item()

        metrics = {'curr_train_loss': loss.detach(),
                  **grad_metrics}

        return logits.detach(), metrics
    
    def compute_activation_statistics(self, batch):
        # Compute the effective feature rank.
        # Compute the number of activations with a value of 0 for all examples in the input batch.
        
        # First do a forward pass.
        with torch.no_grad():
            self.model(batch)
        
        srank_dict = {}
        effective_rank_dict = {}
        dead_neurons_dict = {}

        # Loop through all the activations.
        num_layers = 0.
        total_effective_rank = 0.
        total_srank = 0.

        num_dead_neurons = 0.
        total_neurons = 0.
        for layer_name, activations in self.model.activations.items():
            
            if 'conv' in layer_name:
                batch_size = len(batch)
                activation_matrix = activations.reshape(batch_size, -1)
            else:
                activation_matrix = activations
            
            # Compute the effective rank of the features.
            singular_values = torch.linalg.svdvals(activation_matrix, driver=None, out=None)
            cumulative_fraction = torch.cumsum(singular_values, dim=-1) / torch.sum(singular_values)

            # srank computation is on page 3 of this paper: https://arxiv.org/pdf/2010.14498.pdf 
            delta = 0.01
            srank = len(cumulative_fraction[cumulative_fraction < 1 - delta])
            srank_dict[f'model_feature_srank/{layer_name}'] = srank
            total_srank += srank

            dist = singular_values / torch.sum(singular_values)
            # dist = dist.detach().numpy()
            # entropy = scipy.stats.entropy(dist)
            dist = dist[dist > 0]
            entropy = -1. * torch.sum(dist * torch.log(dist))
            effective_rank = torch.exp(entropy).detach().item()
            effective_rank_dict[f'model_effective_feature_rank/{layer_name}'] = effective_rank
            total_effective_rank += effective_rank
            num_layers += 1
            
            # Count the number of activations which are zero for ALL inputs in the batch.
            num_neurons = activation_matrix.shape[1]
            total_neurons += num_neurons
            
            # activation_matrix is batch_size x hidden dimension for the hidden layer.
            # Compute the number of columns for which all entries are 0.
            is_zero_column = torch.all(activation_matrix == 0, dim=0)
            num_zero_columns = torch.sum(is_zero_column).detach().item()
            num_dead_neurons += num_zero_columns

            fraction_dead_neurons = num_zero_columns / float(num_neurons)
            
            dead_neurons_dict[f'model_dead_neurons_fraction/{layer_name}'] = fraction_dead_neurons

        srank_dict['model_feature_srank/avg_srank'] = total_srank / float(num_layers)
        effective_rank_dict['model_effective_feature_rank/avg_effective_rank'
                            ] = total_effective_rank / float(num_layers)
        dead_neurons_dict['model_dead_neurons_fraction/fraction_dead_neurons'
                          ] = num_dead_neurons / float(total_neurons)
        
        l1_norm_dict = self.model.compute_l1_norm()
        try:
            input_layer_norm_dict = self.model.input_layer_norms()
        except:
            input_layer_norm_dict = None

        if input_layer_norm_dict is not None:
            activation_statistics_dict = {
                **srank_dict,
                **effective_rank_dict,
                **dead_neurons_dict,
                **l1_norm_dict,
                **input_layer_norm_dict,
            }
        else:
            activation_statistics_dict = {
                **srank_dict,
                **effective_rank_dict,
                **dead_neurons_dict,
                **l1_norm_dict,
            }

        return activation_statistics_dict


class LayerNormAgent(BaseAgent):

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device):
        """
        Initializes the LayerNormAgent instance.

        Args:
            model_constructor (callable): A callable that constructs the model. The callable should return an instance of nn.Module.
            optimizer_cfg (dict): Configuration for the optimizer, including optimizer type and hyperparameters.
            loss_fn (callable): The loss function to be used for training. Should accept model predictions and target values as inputs.
            device (torch.device): The device on which the model and computations will run (e.g., 'cpu' or 'cuda').
        """
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)
        # Overwrite the model and optimizer.
        self.model = model_constructor(apply_layer_norm=True)
        self.model.to(self.device)
        self.get_optimizer()


class ConcatReLUAgent(BaseAgent):

    def __init__(self, model_constructor, optimizer_cfg, 
                 loss_fn, device, fraction_to_remove):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)
        # Overwrite the model and optimizer.
        self.model = model_constructor(
            use_crelu=True, 
            fraction_to_remove=fraction_to_remove)
        self.model.to(self.device)
        self.get_optimizer()


class L2Agent(BaseAgent):

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 l2_weight):
        """
        Initializes the L2Agent instance.

        Args:
            model_constructor (callable): A callable that constructs the model. The callable should return an instance of nn.Module.
            optimizer_cfg (dict): Configuration for the optimizer, including optimizer type and hyperparameters.
            loss_fn (callable): The loss function to be used for training. Should accept model predictions and target values as inputs.
            device (torch.device): The device on which the model and computations will run (e.g., 'cpu' or 'cuda').
            l2_weight (float): L2 regularization strength.
        """
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.l2_weight = l2_weight

    def compute_loss(self, x, y):

        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        # Compute the L2 norm.
        l2_loss = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue

            l2_loss += torch.sum(param ** 2)

        loss += self.l2_weight * 0.5 * l2_loss

        return loss, logits


class L2InitAgent(BaseAgent):
    """Computes the L2 distance between the initial and current weights"""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 l2_weight, sample_init_values):
        """
        Initializes the L2InitAgent instance.

        Args:
            model_constructor (callable): A callable that constructs the model. The callable should return an instance of nn.Module.
            optimizer_cfg (dict): Configuration for the optimizer, including optimizer type and hyperparameters.
            loss_fn (callable): The loss function to be used for training. Should accept model predictions and target values as inputs.
            device (torch.device): The device on which the model and computations will run (e.g., 'cpu' or 'cuda').
            l2_weight (float): L2 Init regularization strength.
            sample_init_values (bool): Whether to regularize to fixed or re-sampled initial values.
        """
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.l2_weight = l2_weight
        self.sample_init_values = sample_init_values
        
        self.init_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.init_params_dict[name] = param.data.clone().detach()

    def compute_loss(self, x, y):
        
        if self.sample_init_values:
            init_model_named_params_resampled = self.model_constructor().state_dict()
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        l2_loss = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            
            if self.sample_init_values:
                init_param = init_model_named_params_resampled[name].detach()
            else:
                init_param = self.init_params_dict[name].detach()
            
            diff = param - init_param
            l2_loss += torch.sum(diff ** 2)

        loss += self.l2_weight * 0.5 * l2_loss

        return loss, logits
    

class HuberInitAgent(BaseAgent):
    """Computes the Huber loss between the initial and current weights"""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 huber_weight, sample_init_values):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.huber_weight = huber_weight
        self.sample_init_values = sample_init_values
        
        self.huber_loss_fn = nn.HuberLoss(reduction='sum', delta=1.0)
        
        self.init_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.init_params_dict[name] = param.data.clone().detach()

    def compute_loss(self, x, y):
        
        if self.sample_init_values:
            init_model_named_params_resampled = self.model_constructor().state_dict()
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        huber_loss = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            
            if self.sample_init_values:
                init_param = init_model_named_params_resampled[name].detach()
            else:
                init_param = self.init_params_dict[name].detach()
            
            huber_loss += self.huber_loss_fn(param, init_param)

        loss += self.huber_weight * huber_loss

        return loss, logits


class L1InitAgent(BaseAgent):
    """Computes the L1 distance between the initial and current weights"""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 l1_weight, sample_init_values):
        """
        Initializes the L1InitAgent instance.

        Args:
            model_constructor (callable): A callable that constructs the model. The callable should return an instance of nn.Module.
            optimizer_cfg (dict): Configuration for the optimizer, including optimizer type and hyperparameters.
            loss_fn (callable): The loss function to be used for training. Should accept model predictions and target values as inputs.
            device (torch.device): The device on which the model and computations will run (e.g., 'cpu' or 'cuda').
            l1_weight (float): L1 Init regularization strength.
            sample_init_values (bool): Whether to regularize to fixed or re-sampled initial values.
        """
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.l1_weight = l1_weight
        self.sample_init_values = sample_init_values
        
        self.init_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.init_params_dict[name] = param.data.clone().detach()

    def compute_loss(self, x, y):
        
        if self.sample_init_values:
            init_model_named_params_resampled = self.model_constructor().state_dict()
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        l1_loss = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            
            if self.sample_init_values:
                init_param = init_model_named_params_resampled[name].detach()
            else:
                init_param = self.init_params_dict[name].detach()
            
            diff = param - init_param
            l1_loss += torch.sum(torch.abs(diff))

        loss += self.l1_weight * l1_loss

        return loss, logits
    

class HybridInitAgent(BaseAgent):
    """Computes both the L2 and L1 distances between the initial and current weights"""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 l2_weight, l1_weight, sample_init_values):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.sample_init_values = sample_init_values
        
        self.init_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.init_params_dict[name] = param.data.clone().detach()

    def compute_loss(self, x, y):
        
        if self.sample_init_values:
            init_model_named_params_resampled = self.model_constructor().state_dict()
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        l2_loss = 0
        l1_loss = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            
            if self.sample_init_values:
                init_param = init_model_named_params_resampled[name].detach()
            else:
                init_param = self.init_params_dict[name].detach()
            
            diff = param - init_param
            l2_loss += torch.sum(diff ** 2)
            l1_loss += torch.sum(torch.abs(diff))

        loss += self.l2_weight * 0.5 * l2_loss + self.l1_weight * l1_loss

        return loss, logits
    

class EWCAgent(BaseAgent):
    """Computes the L2 distance between the current weights and past task weights."""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 ewc_weight, use_fisher=False):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.ewc_weight = ewc_weight
        self.use_fisher = use_fisher

        self.star_params_dict = {}
        self.fisher = {}

    def update_star_params(self):
        self.star_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.star_params_dict[name] = param.data.clone().detach()

    # Update the Fisher Information
    def update_fisher(self, xs, ys, batch_size):

        losses = []
        for i in range(0, len(xs) - batch_size, batch_size):
            batch_x = xs[i:i+batch_size]
            batch_y = ys[i:i+batch_size]
            x = batch_x
            y = batch_y

            losses.append(
                F.log_softmax(self.model(x), dim=1)[range(batch_size), y.data]
            )

        # estimate the fisher information of the parameters.
        sample_losses = torch.cat(losses).unbind()
        sample_grads = zip(*[autograd.grad(l, self.model.parameters(), retain_graph=(i < len(sample_losses))) 
        for i, l in enumerate(sample_losses, 1)])

        sample_grads = [torch.stack(gs) for gs in sample_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in sample_grads]
        self.fisher = {}

        for (name, param), fisher in zip(
            self.model.named_parameters(), fisher_diagonals):
            self.fisher[name] = fisher.detach()

    def update_params_and_fisher(self, xs, ys, batch_size):
        self.update_star_params()
        if self.use_fisher:
            self.update_fisher(xs, ys, batch_size)

    def compute_loss(self, x, y):
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        ewc_loss = 0
        if len(self.star_params_dict) > 0:
            for name, param in self.model.named_parameters():
                if not param.requires_grad or 'layer_norm' in name or \
                    'init_params' in name or \
                        'original_last_layer_params' in name:
                    continue
                
                star_param = self.star_params_dict[name].detach()
                
                diff = param - star_param

                fisher = 1
                if self.use_fisher and len(self.fisher):
                    fisher = self.fisher[name]

                ewc_loss += torch.sum(fisher * (diff ** 2)) 

        loss += self.ewc_weight * 0.5 * ewc_loss

        return loss, logits


class L2InitPlusEWCAgent(BaseAgent):
    """Computes the L2 distance between the current weights and past task weights."""

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 l2_weight, ewc_weight, use_fisher):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.l2_weight = l2_weight
        self.ewc_weight = ewc_weight
        self.use_fisher = use_fisher

        self.star_params_dict = {}
        self.fisher = {}

        self.init_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.init_params_dict[name] = param.data.clone().detach()

    def update_star_params(self):
        self.star_params_dict = {}
        # Populate init params dict.
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            self.star_params_dict[name] = param.data.clone().detach()

    # Update the Fisher Information
    def update_fisher(self, xs, ys, batch_size):

        losses = []
        for i in range(0, len(xs) - batch_size, batch_size):
            batch_x = xs[i:i+batch_size]
            batch_y = ys[i:i+batch_size]
            x = batch_x
            y = batch_y

            losses.append(
                F.log_softmax(self.model(x), dim=1)[range(batch_size), y.data]
            )

        # estimate the fisher information of the parameters.
        sample_losses = torch.cat(losses).unbind()
        sample_grads = zip(*[autograd.grad(l, self.model.parameters(), retain_graph=(i < len(sample_losses))) 
        for i, l in enumerate(sample_losses, 1)])

        sample_grads = [torch.stack(gs) for gs in sample_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in sample_grads]
        self.fisher = {}

        for (name, param), fisher in zip(
            self.model.named_parameters(), fisher_diagonals):
            self.fisher[name] = fisher.detach()

    def update_params_and_fisher(self, xs, ys, batch_size):
        self.update_star_params()
        if self.use_fisher:
            self.update_fisher(xs, ys, batch_size)

    def compute_loss(self, x, y):
            
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        l2_loss = 0
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                'init_params' in name or \
                    'original_last_layer_params' in name:
                continue
            
            init_param = self.init_params_dict[name].detach()
            diff = param - init_param
            l2_loss += torch.sum(diff ** 2)

            if len(self.star_params_dict) > 0:
                star_param = self.star_params_dict[name].detach()
                diff = param - star_param
                
                fisher = 1
                if self.use_fisher and len(self.fisher):
                    fisher = self.fisher[name]

                ewc_loss += torch.sum(fisher * (diff ** 2)) 


        loss += self.l2_weight * 0.5 * l2_loss
        loss += self.ewc_weight * 0.5 * ewc_loss

        return loss, logits


class ShrinkAndPerturbAgent(BaseAgent):

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 shrink, perturb_scale):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)

        self.shrink = shrink
        self.perturb_scale = perturb_scale

    def _shrink_and_perturb(self):
        """Shrinks the parameter towards the origin and perturbs it"""

        # Sample perturbation
        random_model = self.model_constructor()

        params = [p for p in self.model.parameters()]
        random_params = [p for p in random_model.parameters()]
        
        with torch.no_grad():
            for param, random_param in zip(params, random_params):
                param.mul_(1. - self.shrink) # Shrink
                param.add_(self.perturb_scale * random_param) # Perturb

    def step(self, x, y):
        logits, metrics = super().step(x, y)
        self._shrink_and_perturb()
        return logits, metrics


class ReDOAgent(BaseAgent):
    """Recycle Dormant Neurons"""    

    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 recycle_period, recycle_threshold):
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)
        
        self.recycle_period = recycle_period
        self.recycle_threshold = recycle_threshold
        self.step_count = 0


        # Populate init params dict.
        self.init_params = []
        self.init_params_dict = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or 'layer_norm' in name or \
                                          'init_params' in name or \
                                          'original_last_layer_params' in name:
                continue

            self.init_params.append(param.data.clone().detach())
            self.init_params_dict[name] = param.data.clone().detach()


    def recycle_neurons(self):

        for layer_id, (layer_type, activation_tuple) in enumerate(self.model.activations_for_redo.items()):
            
            activation_set = activation_tuple[0]
            current_layer_type = activation_tuple[1]
            next_layer_type = activation_tuple[2]
            
            # Compute the expected absolute value activation over the batch.
            expected_activation = torch.mean(torch.abs(activation_set), dim=0)
            
            if current_layer_type == 'conv':
                # Shape (conv layer): batch_size x num_output_channels x feature_map_dim x feature_map_dim
                
                # Compute the expected absolute value activation over the feature map.
                expected_activation = torch.mean(expected_activation, dim=(-2, -1))
            
            # Compute the average expected absolute value activation for the layer.
            average_expected_activation = torch.mean(expected_activation)
            neuron_scores = expected_activation / average_expected_activation
        
            # If neuron score is less than threshold, reset the incoming weights
            # to be the initial values and the outgoing weights to be 0.
            
            for neuron_index in range(len(neuron_scores)):
                if neuron_scores[neuron_index] <= self.recycle_threshold:
                    
                    # Get the incoming and outgoing weight matrices.
                    incoming_weights = getattr(self.model, self.model.layer_names[layer_id])
                    outgoing_weights = getattr(self.model, self.model.layer_names[layer_id + 1])

                    # Reset incoming weights to be initial values.
                    layer_name = self.model.layer_names[layer_id]
                    
                    weight_param_name = f'{layer_name}.weight'
                    initial_weights = self.init_params_dict[weight_param_name]
                    
                    bias_param_name = f'{layer_name}.bias'
                    initial_biases = self.init_params_dict[bias_param_name]

                    with torch.no_grad():
                        incoming_weights.weight.data[neuron_index].copy_(initial_weights.data[neuron_index])
                        incoming_weights.bias.data[neuron_index].copy_(initial_biases.data[neuron_index])
                        
                    if current_layer_type == 'conv' and next_layer_type == 'fc':
                        # Shape of conv activation: (batch_size, output_channels, feature_map_width, feature_map_width)
                        # where feature_map_width is after pooling.
                        # i*(feature_map_width*feature_map_width):(i+1)*(feature_map_width*feature_map_width)
                        
                        # Get the number of channels in the activation.
                        num_channels = activation_set.shape[1]
                        # Get the number of features in the feature map after doing max pool operation.
                        num_features_after_max_pool = int(outgoing_weights.weight.shape[-1] / num_channels)

                        neuron_indices_start = num_features_after_max_pool * neuron_index
                        neuron_indices_end = num_features_after_max_pool * (neuron_index + 1)
                        
                        with torch.no_grad():
                            outgoing_weights.weight.data[:, neuron_indices_start:neuron_indices_end] = 0.
                    else:
                        # Set outgoing weights to be zero.
                        with torch.no_grad():
                            outgoing_weights.weight.data[:, neuron_index] = 0.
        
    def step(self, x, y):

        if self.step_count % self.recycle_period == 0 and self.step_count > 0:
            self.recycle_neurons()

        logits, metrics = super().step(x, y)
        self.step_count += 1
        return logits, metrics

        
class ContinualBackpropAgent(BaseAgent):
    
    def __init__(self, model_constructor, optimizer_cfg, loss_fn, device, 
                 replacement_rate, decay_rate, maturity_threshold, util_type, accumulate):
        
        super().__init__(model_constructor, optimizer_cfg, loss_fn, device)
        
        # Override the optimizer.
        self.get_optimizer()

        if isinstance(self.model, MLP):
            self.gnt = GnT(
                net=self.model.layers,
                hidden_activation='relu',
                opt=self.optimizer,
                replacement_rate=replacement_rate,
                decay_rate=decay_rate,
                maturity_threshold=maturity_threshold,
                util_type=util_type,
                device=self.device,
                loss_func=loss_fn,
                init='kaiming',
                accumulate=accumulate,
            )
        elif isinstance(self.model, ConvNet):
            self.gnt = ConvGnT(
                net=self.model.layers,
                hidden_activation='relu',
                opt=self.optimizer,
                replacement_rate=replacement_rate,
                decay_rate=decay_rate,
                init='kaiming',
                num_last_filter_outputs=self.model.last_filter_output,
                util_type=util_type,
                maturity_threshold=maturity_threshold,
                device=self.device,
            )
            
    def get_optimizer(self):
        if self.optimizer_cfg.name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer_cfg.lr)
        elif self.optimizer_cfg.name == 'Adam':
            self.optimizer = AdamGnT(self.model.parameters(),
                                     lr=self.optimizer_cfg.lr, 
                                     weight_decay=self.optimizer_cfg.weight_decay)

    def step(self, x, y):
        _ = self.predict(x)
        self.previous_features = list(self.model.activations.values())
        logits, metrics = super().step(x, y)
        self.gnt.gen_and_test(features=self.previous_features)
        return logits, metrics
