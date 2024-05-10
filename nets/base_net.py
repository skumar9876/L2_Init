import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def utils_l2_norm(named_params):
    total_norm = 0.0
    for name, param in named_params:
        if 'original_last_layer_params' in name \
            or 'init_params' in name or 'layer_norm' in name:
                continue

        if param.requires_grad:
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm
    
def utils_l1_norm(named_params):
    total_norm = 0
    total_params = 0
    
    l1_norm_dict = {}

    for name, param in named_params:
        if 'original_last_layer_params' in name \
            or 'init_params' in name or 'layer_norm' in name:
                continue
            
        if param.requires_grad:
            l1_norm = param.data.abs().sum().detach().item()
            num_neurons = param.numel()
            
            total_norm += l1_norm
            total_params += num_neurons
            
            l1_norm_dict[f'model_l1_norm/{name}'] = l1_norm / num_neurons

    average_l1_norm = total_norm / total_params
    l1_norm_dict['model_l1_norm/avg_l1_norm'] = average_l1_norm
    
    return l1_norm_dict


class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.init_params = None  # Subclass will overwrite this.
        
    def reset(self):
        self.load_state_dict(copy.deepcopy(self.init_params))
    
    def get_model_weights_l2_norm(self):
        return utils_l2_norm(self.named_parameters())

    def compute_l1_norm(self):
        return utils_l1_norm(self.named_parameters())
    
    def compute_total_params(self):
        # Get the total number of parameters in the neural network
        # NOT including the layer_norm parameters or init params.
        total_params = 0.
        
        for name, param in self.named_parameters():
            if 'layer_norm' not in name and \
                'init_params' not in name and \
                    'original_last_layer_params' not in name:
                    total_params += param.numel()
                    
        return total_params