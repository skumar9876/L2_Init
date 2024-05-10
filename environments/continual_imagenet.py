import random
import numpy as np

import torch 
import torch.nn as nn

from utils.load_data import load_imagenet_data


def generate_class_sequence(num_classes, num_repetitions, rng):
    class_sequence = np.zeros(num_classes * num_repetitions).astype(np.int32)
    for i in range(num_repetitions):
        indices = np.arange(num_classes).astype(np.int32)
        rng.shuffle(indices)
        class_sequence[i * num_classes : (i + 1) * num_classes] = np.arange(num_classes)[indices]
    return np.array(class_sequence)


class ContinualImageNet:
    
    def __init__(self,
                 task_duration=100,
                 num_classes_per_task=2,
                 num_tasks=100,
                 env_batch_size=1, seed=0, device=None):

        self.obs_dim = (3, 32, 32)
    
        self.task_length = task_duration
        self.num_tasks = num_tasks
        self.env_batch_size = env_batch_size
        self.device = device
        
        self.task_type = 'imagenet'
        self.task_type_ids = {}
        self.task_type_ids[self.task_type] = 0
        self.current_task_length = task_duration

        self.num_classes = num_classes_per_task
        self.horizon = self.task_length * self.num_tasks

        self.env_rng = random.Random(seed)
        
        self.index = 0
        self.t = 0
        self.task_id = 0
        
        num_unique_tasks = int(1000 / num_classes_per_task)
        assert num_tasks % num_unique_tasks == 0
        num_repetitions = num_tasks // num_unique_tasks
        self.all_task_pairs = generate_class_sequence(num_classes=1000, 
                                                      num_repetitions=num_repetitions, 
                                                      rng=self.env_rng)
        self.all_task_pairs = np.reshape(self.all_task_pairs, (num_tasks, num_classes_per_task))
            
        self.task_ids_to_train_xs = {}
        self.task_ids_to_train_ys = {}
        self.task_ids_to_test_xs = {}
        self.task_ids_to_test_ys = {}
        
        self._load_task_data(self.task_id)
        self.task_images = self.task_ids_to_train_xs[self.task_id]
        self.task_labels = self.task_ids_to_train_ys[self.task_id]

    def _load_task_data(self, task_id):
        if task_id in self.task_ids_to_train_xs:
            return

        class_labels = self.all_task_pairs[task_id]
        
        task_train_images, task_train_labels, \
            task_test_images, task_test_labels = load_imagenet_data(
                classes=class_labels, rng=self.env_rng)

        dataset_size = len(task_train_images)

        indices = np.arange(dataset_size)
        self.env_rng.shuffle(indices)

        task_train_images = task_train_images[indices]
        task_train_labels = task_train_labels[indices]

        self.task_ids_to_train_xs[task_id] = task_train_images.to(self.device)
        self.task_ids_to_train_ys[task_id] = task_train_labels.to(self.device)
        self.task_ids_to_test_xs[task_id] = task_test_images.to(self.device)
        self.task_ids_to_test_ys[task_id] = task_test_labels.to(self.device)

    def _step(self):
        if self.t > 0 and self.t % self.task_length == 0:
            # Load data for the new task.
            self.task_id += 1
            self.task_images = self.task_ids_to_train_xs[self.task_id]
            self.task_labels = self.task_ids_to_train_ys[self.task_id]
            
            self.index = 0
            self.task_type_ids[self.task_type] += 1

            # Delete old task data that won't be used again.
            # This is to ensure the memory doesn't fill up when running many experiments
            # in parallel.
            self.task_ids_to_train_xs[self.task_id - 2] = []
            self.task_ids_to_train_ys[self.task_id - 2] = []
            self.task_ids_to_test_xs[self.task_id - 2] = []
            self.task_ids_to_test_ys[self.task_id - 2] = []

        elif (self.t + 1) % self.task_length == 0:
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
        
        return curr_x, curr_y
    
    def get_next_sample(self):
        curr_task_timestep = self.t % self.task_length
        x, y = self._step()
        task_id = self.task_id   
        # Indicates whether a new task will begin when calling get_next_sample next. 
        task_done = self.t % self.task_length == 0
        return x, y, task_id, task_done, curr_task_timestep
    
    def get_all_task_data(self, task_id, train=True):
        if task_id not in self.task_ids_to_train_xs:
            self.load_task_data(task_id)
        
        if train:
            xs = self.task_ids_to_train_xs[task_id].detach()
            ys = self.task_ids_to_train_ys[task_id].detach()
        else:
            xs = self.task_ids_to_test_xs[task_id].detach()
            ys = self.task_ids_to_test_ys[task_id].detach()
        
        return xs, ys