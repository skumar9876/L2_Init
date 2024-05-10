import itertools
import numpy as np
import random
from tqdm import tqdm, trange

import torch 
import torch.nn as nn

from utils.load_data import load_cifar100_data


def generate_class_sequence(num_classes, num_repetitions, rng):
    class_sequence = np.zeros(num_classes * num_repetitions).astype(np.int32)
    for i in range(num_repetitions):
        indices = np.arange(num_classes).astype(np.int32)
        rng.shuffle(indices)
        class_sequence[i * num_classes : (i + 1) * num_classes] = np.arange(num_classes)[indices]
    return np.array(class_sequence)

class EasyHardCIFAR100:
    
    def __init__(self,
                 easy_task_duration=1000,
                 hard_task_duration=1000,
                 easy_task_num_classes=1,
                 hard_task_num_classes=5,
                 env_batch_size=1,
                 seed=0,
                 device=None):
        
        self.easy_task_num_classes = easy_task_num_classes
        self.hard_task_num_classes = hard_task_num_classes

        self.env_batch_size = env_batch_size
        self.task_id_to_train_xs = {}
        self.task_id_to_train_ys = {}
        self.task_id_to_test_xs = {}
        self.task_id_to_test_ys = {}
        self.device = device

        self.env_rng = random.Random(seed)
        
        self.obs_dim = (3, 32, 32)
        self.num_classes = 100

        self.index = 0
        self.t = 0
        self.task_id = 0
        
        self.task_types = ['hard', 'easy']
        self.task_type_counters = {}
        self.task_type_durations = {}
        self.task_type_ids = {}
        self.task_type_durations['hard'] = hard_task_duration
        self.task_type_durations['easy'] = easy_task_duration
        self.task_type_index = 0
        self.task_type = self.task_types[self.task_type_index]
        self.current_task_length = self.task_type_durations[self.task_type]
        for task_type in self.task_types:
            self.task_type_ids[task_type] = 0
            self.task_type_counters[task_type] = 0

        # Load and shuffle data.
        self.train_images, self.train_labels, \
            self.test_images, self.test_labels = load_cifar100_data(rng=self.env_rng)
        
        cifar_class_sequence = generate_class_sequence(100, num_repetitions=1, rng=self.env_rng)
        
        self.cifar_task_classes = []
        index = 0
        counter = 0
        done = False
        while not done:
            if counter % 2 == 0:
                self.cifar_task_classes.append(cifar_class_sequence[index:index+self.hard_task_num_classes])
                index += self.hard_task_num_classes
            else:
                self.cifar_task_classes.append(np.array(cifar_class_sequence[index:index+self.easy_task_num_classes]))
                index += self.easy_task_num_classes
            
            if counter % 2 == 0 and index + self.easy_task_num_classes > len(cifar_class_sequence):
                done = True
            elif index + self.hard_task_num_classes > len(cifar_class_sequence):
                done = True
            else:
                counter += 1
        
        self.num_tasks = len(self.cifar_task_classes) - 1
        self.horizon = int(0.5 * (easy_task_duration + hard_task_duration) * self.num_tasks) 
            
        self.train_images.to(device)
        self.train_labels.to(device)
        self.test_images.to(device)
        self.test_labels.to(device)

        self._load_task_data(self.task_id)
        
        self.task_images = self.task_id_to_train_xs[self.task_id]
        self.task_labels = self.task_id_to_train_ys[self.task_id]

    def _load_task_data(self, task_id):
        
        # Check if task data previously loaded.
        if task_id in self.task_id_to_train_xs:
            return

        classes = self.cifar_task_classes[task_id]
        
        if task_id % 2 == 0:
            task_train_images_sequence = []
            task_train_labels_sequence = []
            task_test_images_sequence = []
            task_test_labels_sequence = []
            
            for i, label in enumerate(classes):
                task_train_images_sequence.append(self.train_images[self.train_labels == label]) 
                task_train_labels_sequence.append(self.train_labels[self.train_labels == label]) 
                task_test_images_sequence.append(self.test_images[self.test_labels == label]) 
                task_test_labels_sequence.append(self.test_labels[self.test_labels == label]) 
                
            task_train_images = torch.cat(task_train_images_sequence, dim=0)
            task_train_labels = torch.cat(task_train_labels_sequence, dim=0)
            task_test_images = torch.cat(task_test_images_sequence, dim=0)
            task_test_labels = torch.cat(task_test_labels_sequence, dim=0)
        else:
            label = classes[0]
            task_train_images = self.train_images[self.train_labels == label]
            task_train_labels = self.train_labels[self.train_labels == label]
            task_test_images = self.test_images[self.test_labels == label]
            task_test_labels = self.test_labels[self.test_labels == label]

        dataset_size = len(task_train_images)
        
        indices = np.arange(dataset_size)
        self.env_rng.shuffle(indices)
        
        task_train_images = task_train_images[indices]
        task_train_labels = task_train_labels[indices]

        self.task_id_to_train_xs[task_id] = task_train_images.to(self.device)
        self.task_id_to_train_ys[task_id] = task_train_labels.to(self.device)
        self.task_id_to_test_xs[task_id] = task_test_images.to(self.device)
        self.task_id_to_test_ys[task_id] = task_test_labels.to(self.device)

    def _step(self):
        if self.t > 0 and self.task_type_counters[self.task_type] == self.task_type_durations[self.task_type]:
            
            # Load data for the new task.
            
            # Reset counter for finished task type to be 0.
            self.task_type_counters[self.task_type] = 0
            # Increment task id for finished task type.
            self.task_type_ids[self.task_type] += 1
            # Increment task id.
            self.task_id += 1

            self.task_type_index = (self.task_type_index + 1) % 2
            self.task_type = self.task_types[self.task_type_index]
            self.current_task_length = self.task_type_durations[self.task_type]
            
            self.task_images = self.task_id_to_train_xs[self.task_id]
            self.task_labels = self.task_id_to_train_ys[self.task_id]
            
            self.index = 0

            # Hack to delete old permutation data that won't be used again.
            # This is to ensure the memory doesn't fill up when running many experiments
            # in parallel.
            self.task_id_to_train_xs[self.task_id - 2] = []
            self.task_id_to_train_ys[self.task_id - 2] = []
            self.task_id_to_test_xs[self.task_id - 2] = []
            self.task_id_to_test_ys[self.task_id - 2] = []

        elif self.task_type_counters[self.task_type] + 1 == self.task_type_durations[self.task_type]:
            # Generate data for the next task.
            next_task_id = self.task_id + 1
            self._load_task_data(next_task_id)
        
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
        self.task_type_counters[self.task_type] += 1
        
        return curr_x, curr_y

    def get_next_sample(self):
        curr_task_timestep = self.task_type_counters[self.task_type]
        x, y = self._step()
        task_id = self.task_id   
        # Indicates whether a new task will begin when calling get_next_sample next. 
        task_done = self.task_type_counters[self.task_type] == self.task_type_durations[self.task_type]
        return x, y, task_id, task_done, curr_task_timestep
    
    def get_all_task_data(self, task_id, train=True):
        if task_id not in self.task_id_to_train_xs:
            self.load_task_data(task_id)
        
        if train:
            xs = self.task_id_to_train_xs[task_id].detach()
            ys = self.task_id_to_train_ys[task_id].detach()
        else:
            xs = self.task_id_to_test_xs[task_id].detach()
            ys = self.task_id_to_test_ys[task_id].detach()
        
        return xs, ys