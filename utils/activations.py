import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspired from Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units
# https://arxiv.org/pdf/1603.05201.pdf

class CReLU(nn.Module):

    def __init__(self, inplace=False):
        super(CReLU, self).__init__()

    def forward(self, x):
        x = torch.cat((x,-x),-1)
        return F.relu(x)
    
    
class ConvCReLU(nn.Module):

    def __init__(self, inplace=False):
        super(ConvCReLU, self).__init__()

    def forward(self, x):
        # Concatenate along the channel dimension.
        channel_dim = 1
        x = torch.cat((x,-x), channel_dim)
        return F.relu(x)