import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.base_net import BaseNet
from utils.activations import CReLU


class MLP(BaseNet):
    def __init__(self, input_size=784, num_hidden=2, num_classes=10, hidden_size=10, 
                 num_channels=None,
                 apply_layer_norm=False, use_crelu=False,
                 fraction_to_remove=0.0,
                 init_type='default'):
        super().__init__()
        
        del num_channels
        
        self.num_hidden = num_hidden
        self.use_crelu = use_crelu
        
        if self.use_crelu:
            hidden_size = int((1. - fraction_to_remove) * hidden_size)
        

        out_hidden_size = hidden_size
        in_hidden_size = hidden_size * (2 if self.use_crelu else 1)
        self.activation_fn = CReLU() if self.use_crelu else F.relu

        self.input_layer = nn.Linear(input_size, out_hidden_size)
        
        if self.num_hidden == 2:
            self.fc2 = nn.Linear(in_hidden_size, out_hidden_size)
            self.layer_names = ['input_layer', 'fc2', 'output_layer']
            self.hidden_layers = [self.input_layer, self.fc2]
            self.layers = [self.input_layer, None, self.fc2, None]
        else:
            raise AssertionError(f'Number of hidden layers = {self.num_hidden} not supported.')
        
        self.output_layer = nn.Linear(in_hidden_size, num_classes)
        self.layers.append(self.output_layer)


        # Initialize all layers.
        assert init_type in ['default', 
                             'kaiming_uniform', 'kaiming_normal',
                             'xavier_uniform', 'xavier_normal']
        self.init_type = init_type
        
        with torch.no_grad():
            for layer_index, layer in enumerate(self.layers):
                if layer is None:
                    continue

                gain = nn.init.calculate_gain('relu')
                if layer_index == len(self.layers) - 1: gain = 1.0

                if self.init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    layer.bias *= 0
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight, gain=gain)
                    layer.bias *= 0
                elif self.init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    layer.bias *= 0
                elif self.init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    layer.bias *= 0

        
        self.apply_layer_norm = apply_layer_norm
        if self.apply_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)
            self.layer_norms = [self.layer_norm1, self.layer_norm2]            
            
        self.init_params = copy.deepcopy(self.state_dict())

    def forward(self, x):
        self.activations = {}
        self.activations_for_redo = {}

        for i in range(self.num_hidden):
            x = self.hidden_layers[i](x)
            if self.apply_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation_fn(x)    
            self.activations[self.layer_names[i]] = x
            self.activations_for_redo[self.layer_names[i]] = (x, 'fc', 'fc')

        x = self.output_layer(x)
            
        return x
    
    
    def input_layer_norms(self):
        named_params = self.named_parameters()
        total_norm = 0
        total_params = 0
        
        l1_norm_dict = {}

        for name, param in named_params:
            if 'input_layer' not in name or 'weight' not in name: continue
                
            if param.requires_grad:
                num_inputs = len(param.data[0])
                l1_norm_first_half = param.data[:, :num_inputs // 2].abs().sum().detach().item()
                l1_norm_second_half = param.data[:, num_inputs // 2:].abs().sum().detach().item()
                num_neurons = param.numel() // 2
                
                l1_norm_dict[f'model_l1_norm/{name}_first_half'] = l1_norm_first_half / num_neurons
                l1_norm_dict[f'model_l1_norm/{name}_second_half'] = l1_norm_second_half / num_neurons
        
        return l1_norm_dict

