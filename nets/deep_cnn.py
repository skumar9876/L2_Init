import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.base_net import BaseNet
from utils.activations import CReLU, ConvCReLU

class DeepConvNet(BaseNet):
    def __init__(self, input_size=None, num_hidden=None, hidden_size=None, num_channels=16,
                 num_classes=10, apply_layer_norm=False, use_crelu=False, fraction_to_remove=0):
        super().__init__()

        del input_size  # unused
        del num_hidden # unused
        self.use_crelu = use_crelu
        
        if self.use_crelu:
            hidden_size = int((1. - fraction_to_remove) * hidden_size)
            num_channels = int((1. - fraction_to_remove) * num_channels)
        
        in_hidden_size = hidden_size
        out_hidden_size = hidden_size
        in_channels = num_channels
        out_channels = num_channels

        self.conv_activation_fn = F.relu
        self.fc_activation_fn = F.relu        
        if self.use_crelu:
            in_hidden_size *= 2
            in_channels *= 2
            self.conv_activation_fn = ConvCReLU()
            self.fc_activation_fn = CReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=5)
        output_shape1 = (in_channels, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        output_shape2 = (in_channels, 12, 12)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        output_shape3 = (in_channels, 4, 4)

        # 5 by 5 here because of max pool being applied.
        self.last_filter_output = 5 * 5
        
        flattened_dim = 2 * 2 * in_channels

        # flattened_dim = in_channels * self.last_filter_output
        self.fc1 = nn.Linear(flattened_dim, out_hidden_size)
        self.fc2 = nn.Linear(in_hidden_size, out_hidden_size)
        self.fc3 = nn.Linear(in_hidden_size, out_hidden_size)
        self.output_layer = nn.Linear(in_hidden_size, num_classes)

        self.conv_layer_names = ['conv1', 'conv2', 'conv3']
        self.fc_layer_names = ['fc1', 'fc2', 'fc3', 'output_layer']
        self.layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3', 'output_layer']
        
        self.init_params = copy.deepcopy(self.state_dict())
        
        self.layers = [self.conv1, None, self.conv2, None, self.conv3, None, 
                       self.fc1, None, self.fc2, None, self.fc3, None, self.output_layer]
        
        self.apply_layer_norm = apply_layer_norm
        if self.apply_layer_norm:
            self.layer_norm1 = nn.LayerNorm(output_shape1)
            self.layer_norm2 = nn.LayerNorm(output_shape2)
            self.layer_norm3 = nn.LayerNorm(output_shape3)
            self.layer_norm4 = nn.LayerNorm(in_hidden_size)
            self.layer_norm5 = nn.LayerNorm(in_hidden_size)
            self.layer_norm6 = nn.LayerNorm(in_hidden_size)
            self.layer_norms = [self.layer_norm1, self.layer_norm2, 
                                self.layer_norm3, self.layer_norm4, 
                                self.layer_norm5, self.layer_norm6]

    def forward(self, x):
        self.previous_layer_type = {}
        self.next_layer_type = {}
        self.activations_for_redo = {}
        
        self.activations = {}
        x = self.conv1(x)
        if self.apply_layer_norm:
            x = self.layer_norm1(x)
        x = self.conv_activation_fn(x)
        self.activations_for_redo['conv1'] = (x, 'conv', 'conv')
        x = self.pool(x)
        self.activations['conv1'] = x       
 
        x = self.conv2(x)        
        if self.apply_layer_norm:
            x = self.layer_norm2(x)
        x = self.conv_activation_fn(x)
        self.activations_for_redo['conv2'] = (x, 'conv', 'fc')
        x = self.pool(x)
        self.activations['conv2'] = x
        
        x = self.conv3(x)        
        if self.apply_layer_norm:
            x = self.layer_norm3(x)
        x = self.conv_activation_fn(x)
        self.activations_for_redo['conv3'] = (x, 'conv', 'fc')
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        self.activations['conv3'] = x

        x = self.fc1(x)
        if self.apply_layer_norm:
            x = self.layer_norm4(x)
        x = self.fc_activation_fn(x)
        self.activations_for_redo['fc1'] = (x, 'fc', 'fc')
        self.activations['fc1'] = x
        
        x = self.fc2(x)
        if self.apply_layer_norm:
            x = self.layer_norm5(x)
        x = self.fc_activation_fn(x)
        self.activations_for_redo['fc2'] = (x, 'fc', 'fc')
        self.activations['fc2'] = x
        
        x = self.fc3(x)
        if self.apply_layer_norm:
            x = self.layer_norm6(x)
        x = self.fc_activation_fn(x)
        self.activations_for_redo['fc3'] = (x, 'fc', 'fc')
        self.activations['fc3'] = x

        x = self.output_layer(x)  
        return x