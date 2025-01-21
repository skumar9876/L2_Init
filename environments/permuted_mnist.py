import itertools
import numpy as np
import random
from tqdm import tqdm, trange

import torch 
import torch.nn as nn

from utils.load_data import load_mnist_data


# Define function to generate a permutation of the input pixels
def generate_permutation(train_images, test_images, rng):
    assert len(train_images.shape) == 2

    # Generate permuted indices.
    num_pixels = train_images.shape[1]
    permuted_indices = np.arange(num_pixels)
    rng.shuffle(permuted_indices)

    # Generate permuted images.
    train_permuted_images = train_images[:, permuted_indices]
    test_permuted_images = test_images[:, permuted_indices]

    return train_permuted_images, test_permuted_images


class PermutedMNIST:
    
    def __init__(self,
                 permutation_duration=60000,
                 num_permutations=10,
                 env_batch_size=1, unique_samples_per_dataset=60000,
                 seed=0,
                 device=None):

        self.permutation_duration = permutation_duration
        self.task_length = permutation_duration
        self.num_permutations = num_permutations
        self.env_batch_size = env_batch_size
        self.unique_samples_per_dataset = unique_samples_per_dataset
        self.permutations_to_train_xs = {}
        self.permutations_to_train_ys = {}
        self.permutations_to_test_xs = {}
        self.permutations_to_test_ys = {}
        self.device = device

        self.env_rng = random.Random(seed)
        
        self.obs_dim = (784,)
        self.num_classes = 10
        self.horizon = self.permutation_duration * self.num_permutations

        self.index = 0
        self.t = 0
        self.task_id = 0
        
        self.task_type = 'imagenet'
        self.task_type_ids = {}
        self.task_type_ids[self.task_type] = 0
        self.current_task_length = self.permutation_duration

        # Load and shuffle data.
        self.train_images, self.train_labels, \
            self.test_images, self.test_labels = load_mnist_data(rng=self.env_rng)
            
        # Use only 1000 images for testing.
        self.test_images = self.test_images[:1000]
        self.test_labels = self.test_labels[:1000]

        self.train_images = np.reshape(self.train_images, (len(self.train_images), -1))
        self.test_images = np.reshape(self.test_images, (len(self.test_images), -1))
            
        self.train_images = self.train_images.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_images = self.test_images.to(device)
        self.test_labels = self.test_labels.to(device)
        
        self.train_images = self.train_images.detach()
        self.train_labels = self.train_labels.detach()
        self.test_images = self.test_images.detach()
        self.test_labels = self.test_labels.detach()

        self._load_permutation_data(self.task_id)
        self.task_images = self.permutations_to_train_xs[self.task_id]
        self.task_labels = self.permutations_to_train_ys[self.task_id]

    def _load_permutation_data(self, permutation):
        
        # Check if permutation data previously loaded.
        if permutation in self.permutations_to_train_xs:
            return
        
        # Construct data for new permutation.
        if permutation == 0:
            task_train_images = self.train_images
            task_test_images = self.test_images
        else:
            task_train_images, task_test_images = generate_permutation(
                self.train_images, self.test_images, self.env_rng)
        
        task_train_labels = self.train_labels
        task_test_labels = self.test_labels

        dataset_size = len(task_train_images)
        
        indices = np.arange(dataset_size)
        self.env_rng.shuffle(indices)
        indices = indices[:self.unique_samples_per_dataset]
        
        task_train_images = task_train_images[indices]
        task_train_labels = task_train_labels[indices]

        self.permutations_to_train_xs[permutation] = task_train_images.to(self.device)
        self.permutations_to_train_ys[permutation] = task_train_labels.to(self.device)
        self.permutations_to_test_xs[permutation] = task_test_images.to(self.device)
        self.permutations_to_test_ys[permutation] = task_test_labels.to(self.device)

    def _step(self):
        if self.t > 0 and self.t % self.permutation_duration == 0:
            # Load data for the new task.
            self.task_id += 1
            self.task_images = self.permutations_to_train_xs[self.task_id]
            self.task_labels = self.permutations_to_train_ys[self.task_id]
            
            self.task_type_ids[self.task_type] += 1
            
            self.index = 0

            # Hack to delete old permutation data that won't be used again.
            # This is to ensure the memory doesn't fill up when running many experiments
            # in parallel.
            # TODO: Uncomment below.
            # self.permutations_to_train_xs[self.task_id - 2] = []
            # self.permutations_to_train_ys[self.task_id - 2] = []
            # self.permutations_to_test_xs[self.task_id - 2] = []
            # self.permutations_to_test_ys[self.task_id - 2] = []

        elif (self.t + 1) % self.permutation_duration == 0:
            # Generate data for the next task.
            next_task_id = self.task_id + 1
            self._load_permutation_data(next_task_id)
        
        if self.index + self.env_batch_size >= len(self.task_images):
            self.index = 0

        if self.index == 0:
            # Shuffle task data when an epoch through all the current task data has finished.
            shuffled_indices = np.arange(len(self.task_labels))
            self.env_rng.shuffle(shuffled_indices)
            self.task_images = self.task_images[shuffled_indices]
            self.task_labels = self.task_labels[shuffled_indices]

        curr_x = self.task_images[self.index:self.index + self.env_batch_size]
        curr_y = self.task_labels[self.index:self.index + self.env_batch_size]
        
        self.index += self.env_batch_size
        self.index %= len(self.task_images)
        
        self.t += 1
        
        return curr_x, curr_y

    def get_next_sample(self):
        curr_task_timestep = self.t % self.permutation_duration
        x, y = self._step()
        task_id = self.task_id   
        # Indicates whether a new task will begin when calling get_next_sample next. 
        task_done = self.t % self.permutation_duration == 0
        return x, y, task_id, task_done, curr_task_timestep
    
    def get_all_task_data(self, task_id, train=True):
        if task_id not in self.permutations_to_train_xs:
            self.load_permutation_data(task_id)
        
        if train:
            xs = self.permutations_to_train_xs[task_id].detach()
            ys = self.permutations_to_train_ys[task_id].detach()
        else:
            xs = self.permutations_to_test_xs[task_id].detach()
            ys = self.permutations_to_test_ys[task_id].detach()
        
        return xs, ys