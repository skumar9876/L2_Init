import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


def shuffle_data(images, labels, rng):
    shuffled_indices = np.arange(len(images))
    rng.shuffle(shuffled_indices)
    return images[shuffled_indices], labels[shuffled_indices]


# Load MNIST data.
def load_mnist_data(rng, resize=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    if resize is not None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((resize, resize)),
                                        transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data.
    trainset = torchvision.datasets.MNIST(
        '~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    # Get all train images and labels.
    # train_images, train_labels = trainset.data[0], trainset.targets[0]
    train_images, train_labels = next(iter(trainloader))
    train_images, train_labels = shuffle_data(train_images, train_labels, rng)
    
    # Download and load the test data.
    testset = torchvision.datasets.MNIST(
        '~/.pytoch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    # Get all test images and labels.
    # test_images, test_labels = testset.data[0], testset.targets[0]
    test_images, test_labels = next(iter(testloader))
    test_images, test_labels = shuffle_data(test_images, test_labels, rng)
    
    return train_images, train_labels, test_images, test_labels


# Load MNIST100 data.
def load_mnist100_data(rng, resize=None):
    """Creates an mnist dataset with 100 classes where each image is a concatenation of two MNIST images. 
       Thus, the images are 01,02,...,99
    """
    train_images, train_labels, test_images, test_labels = load_mnist_data(rng, resize=resize)

    new_train_images = []
    new_train_labels = []
    new_test_images = []
    new_test_labels = []

    for class1 in range(10):
        for class2 in range(10):

            # shape: (num_images, 1, 28, 28)
            class1_train_images = train_images[train_labels == class1]
            class2_train_images = train_images[train_labels == class2]

            class1_test_images = test_images[test_labels == class1]
            class2_test_images = test_images[test_labels == class2]
            
            train_length = 5000
            
            class1_train_images = class1_train_images[:train_length]
            class2_train_images  = class2_train_images[:train_length]
            
            test_length = 500

            class1_test_images = class1_test_images[:test_length]
            class2_test_images = class2_test_images[:test_length]

            # Some sanity-checks
            assert class1_train_images.shape == class2_train_images.shape
            assert class1_test_images.shape == class2_test_images.shape

            # Vertical concatenation
            new_train_images_batch = torch.cat([class1_train_images, class2_train_images], dim=-2) 
            new_test_images_batch = torch.cat([class1_test_images, class2_test_images], dim=-2)

            # Create label
            new_class = class1 * 10 + class2

            # Append everything
            new_train_images.append(new_train_images_batch)
            new_train_labels.append([new_class] * len(class1_train_images))

            new_test_images.append(new_test_images_batch)
            new_test_labels.append([new_class] * len(class1_test_images))


    # Concatenate
    new_train_images = torch.cat(new_train_images)
    new_train_labels = torch.from_numpy(np.array(new_train_labels).flatten()).long()
    new_test_images = torch.cat(new_test_images)
    new_test_labels = torch.from_numpy(np.array(new_test_labels).flatten()).long()

    return new_train_images, new_train_labels, new_test_images, new_test_labels


# Load CIFAR data.
def load_cifar_data(rng):
    transform = transforms.Compose(
               [transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # Normalization values from here: https://github.com/kuangliu/pytorch-cifar/issues/19

    # Download and load the training data.   
    trainset = torchvision.datasets.CIFAR10(root='./cifar-10-data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    
    # Get all train images and labels.
    # train_images, train_labels = trainset.data[0], trainset.targets[0]
    train_images, train_labels = next(iter(trainloader))
    train_images, train_labels = shuffle_data(train_images, train_labels, rng)
    
    # Download and load the test data.
    testset = torchvision.datasets.CIFAR10(root='./cifar-10-data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    # Get all test images and labels.
    # test_images, test_labels = testset.data[0], testset.targets[0]
    test_images, test_labels = next(iter(testloader))
    test_images, test_labels = shuffle_data(test_images, test_labels, rng)
    
    return train_images, train_labels, test_images, test_labels


# Load CIFAR100 data.
def load_cifar100_data(rng):
    transform = transforms.Compose(
               [transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # Normalization values from here: https://github.com/kuangliu/pytorch-cifar/issues/19

    # Download and load the training data.   
    trainset = torchvision.datasets.CIFAR100(root='./cifar-100-data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    
    # Get all train images and labels.
    # train_images, train_labels = trainset.data[0], trainset.targets[0]
    train_images, train_labels = next(iter(trainloader))
    train_images, train_labels = shuffle_data(train_images, train_labels, rng)
    
    # Download and load the test data.
    testset = torchvision.datasets.CIFAR100(root='./cifar-100-data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    # Get all test images and labels.
    # test_images, test_labels = testset.data[0], testset.targets[0]
    test_images, test_labels = next(iter(testloader))
    test_images, test_labels = shuffle_data(test_images, test_labels, rng)
    
    return train_images, train_labels, test_images, test_labels


# Load ImageNet data.
def load_imagenet_data(classes=np.arange(2), rng=None):
    train_images_per_class = 600
    test_images_per_class = 100
    
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'utils/imagenet_data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    
    x_train, y_train = shuffle_data(x_train, y_train, rng)
    x_test, y_test = shuffle_data(x_test, y_test, rng)
    
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)

    return x_train, y_train, x_test, y_test


def unnormalize(xs):
    min_val = np.min(xs, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
    new_xs = (xs - min_val)
    max_val = np.max(new_xs, axis=(0, 2, 3)).reshape(1, 3, 1, 1)
    new_xs = new_xs / max_val
    new_xs = np.transpose(new_xs, (0, 2, 3, 1))
    return new_xs