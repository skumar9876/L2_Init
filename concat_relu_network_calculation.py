from nets import MLP, ConvNet, DeepConvNet
import numpy as np


def compute_fraction_to_cut(
    network_type, 
    num_channels=None,
    output_dim=None, 
    hidden_size=None,
    num_hidden=None):
    
    # Step 1. Compute the number of parameters in the given network architecture.
    if network_type == 'mlp':
        original_net = MLP(num_hidden=num_hidden, 
                           hidden_size=hidden_size, 
                           num_classes=output_dim,
                           use_crelu=False)
    elif network_type == 'cnn':
        original_net = ConvNet(num_channels=num_channels, 
                               hidden_size=hidden_size,
                               num_classes=output_dim,
                               use_crelu=False)
    elif network_type == 'deep_cnn':
        original_net = DeepConvNet(num_channels=num_channels, 
                                   hidden_size=hidden_size,
                                   num_classes=output_dim,
                                   use_crelu=False)
        
    original_num_params = original_net.compute_total_params()

    # Step 2. Compute the number of paramters after a fixed fraction is removed 
    #         from all hidden layers.
    selected_fraction = None
    crelu_num_params = None
    
    
    for fraction in [0.5 - i for i in np.arange(0.0, 0.5, 0.01)]:
        if network_type == 'mlp':
            candidate_crelu_net = MLP(
                num_hidden=num_hidden, hidden_size=hidden_size, num_classes=output_dim, 
                use_crelu=True, fraction_to_remove=fraction)
        elif network_type == 'cnn':
            candidate_crelu_net = ConvNet(
                num_channels=num_channels,
                num_hidden=num_hidden, hidden_size=hidden_size, 
                num_classes=output_dim,
                use_crelu=True, fraction_to_remove=fraction)
        elif network_type == 'deep_cnn':
            candidate_crelu_net = DeepConvNet(num_channels=num_channels, 
                                              num_hidden=num_hidden, 
                                              hidden_size=hidden_size, 
                                              num_classes=output_dim,
                                              use_crelu=True, fraction_to_remove=fraction)
        
        candidate_num_params = candidate_crelu_net.compute_total_params()
        # print(fraction, candidate_num_params)

        if candidate_num_params >= original_num_params:
            selected_fraction = fraction
            crelu_num_params = candidate_num_params
            break
    
    print()
    
    return {
        'selected_fraction': selected_fraction,
        'original_num_params': original_num_params,
        'crelu_num_params': crelu_num_params
    }
    
    
print('MLP environments - Permuted MNIST and Random Label MNIST')
print(f'Hidden dimension: {400}')
print(f'Output dimension: {10}')
print(f'Num hidden: {2}')
print()


result_dict = compute_fraction_to_cut(
    network_type='mlp', 
    num_hidden=2,
    hidden_size=400,
    output_dim=10)

for key in result_dict:
    print(key, result_dict[key])
    
    
print()
print()
print()


print('ConvNet environments - Continual Imagenet')
print(f'Num channels: {64}')
print(f'Hidden dimension: {400}')
print(f'Output dimension: {2}')
print(f'Num hidden: {2}')
print()


result_dict = compute_fraction_to_cut(
    network_type='cnn', 
    num_hidden=2,
    num_channels=64,
    hidden_size=400,
    output_dim=2)

for key in result_dict:
    print(key, result_dict[key])   


print()
print()
print()


print('ConvNet environments - 5+1 CIFAR100')
print(f'Num channels: {64}')
print(f'Hidden dimension: {400}')
print(f'Output dimension: {100}')
print(f'Num hidden: {2}')
print()


result_dict = compute_fraction_to_cut(
    network_type='cnn', 
    num_hidden=2,
    num_channels=64,
    hidden_size=400,
    output_dim=100)

for key in result_dict:
    print(key, result_dict[key])  
    
    
print()
print()
print()

print('ConvNet environments - RandomLabelCIFAR with hidden size = 100')
print(f'Num channels: {64}')
print(f'Hidden dimension: {400}')
print(f'Output dimension: {10}')
print(f'Num hidden: {2}')
print()


result_dict = compute_fraction_to_cut(
    network_type='cnn', 
    num_hidden=2,
    num_channels=64,
    hidden_size=400,
    output_dim=10)

for key in result_dict:
    print(key, result_dict[key])  
    
print()
print()
print()
    
    
print('DeepConvNet environments - RandomLabelCIFAR with hidden size = 100')
print(f'Num channels: {16}')
print(f'Hidden dimension: {100}')
print(f'Output dimension: {10}')
print(f'Num hidden: {3}')
print()


result_dict = compute_fraction_to_cut(
    network_type='deep_cnn', 
    num_hidden=2,
    num_channels=16,
    hidden_size=100,
    output_dim=10)

for key in result_dict:
    print(key, result_dict[key])  