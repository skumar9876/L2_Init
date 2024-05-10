import random
import numpy as np

import torch 
import torch.nn as nn

from utils.load_data import load_mnist_data, load_cifar_data


class RandomLabelEnv:
    def __init__(
        self, 
        concept_duration,
        num_concept_shifts,
        num_labels_changed=10,
        env_batch_size=16, 
        unique_samples_per_dataset=60000, 
        seed=0,
        device=None):
        """
        concept_duration: int, number of time steps after which concept shifts.
        num_concept_shifts: int, number of times that the concept shifts.
        number_of_labels_changed: int, number of labels changed when a concept shift occurs.
        env_batch_size: int
        unique_samples_per_dataset: int
        """
        assert num_labels_changed >= 1 and num_labels_changed <= 10
        assert unique_samples_per_dataset % env_batch_size == 0

        self.task_type = 'random_label'
        self.task_type_ids = {}
        self.task_type_ids[self.task_type] = 0
        self.current_task_length = concept_duration
        self.concept_duration = concept_duration
        self.task_length = concept_duration
        self.num_concept_shifts = num_concept_shifts
        self.num_labels_changed = num_labels_changed
        self.env_batch_size = env_batch_size
        self.unique_samples_per_dataset = unique_samples_per_dataset
        self.horizon = self.concept_duration * self.num_concept_shifts
        
        self.env_rng = random.Random(seed)
        
        self.images = None
        self.labels = None
        self.task_images = None
        self.task_labels = None
        
        self.index = 0
        self.t = 0
        self.task_id = 0

        self.task_id_to_new_labels = {}
    
    def _step(self):
        if self.t > 0 and self.t % self.concept_duration == 0:
            # Load data for the new task.
            self.task_id += 1
            self.task_images = self.images[:]
            self.task_labels = self.task_id_to_new_labels[self.task_id][:]

            self.index = 0
            self.task_type_ids[self.task_type] += 1

            # Delete old task label information that won't be used again.
            if self.task_id - 2 in self.task_id_to_new_labels:
                del self.task_id_to_new_labels[self.task_id - 2]

        elif (self.t + 1) % self.concept_duration == 0:
            # Generate data for the next task.
            all_indices = np.arange(len(self.labels))
            self.env_rng.shuffle(all_indices)
            self.task_id_to_new_labels[self.task_id + 1] = self.labels[all_indices]

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
        
    def get_task_length(self):
        return self.concept_duration

    def get_horizon(self):
        return self.horizon
    
    def get_all_task_data(self, task_id, train=True):
        images = self.images.detach()
        labels = self.task_id_to_new_labels[task_id].detach()
        return images, labels 
    
    def get_next_sample(self):
        curr_task_timestep = self.t % self.concept_duration
        x, y = self._step()
        task_id = self.task_id   
        # Indicates whether a new task will begin when calling get_next_sample next. 
        task_done = self.t % self.concept_duration  == 0
        return x, y, task_id, task_done, curr_task_timestep


class RandomLabelMNIST(RandomLabelEnv):
    def __init__(
        self, 
        concept_duration,
        num_concept_shifts,
        num_labels_changed=10,
        env_batch_size=16, 
        unique_samples_per_dataset=60000, 
        seed=0,
        device=None):
        """
        concept_duration: int, number of time steps after which concept shifts.
        num_concept_shifts: int, number of times that the concept shifts.
        number_of_labels_changed: int, number of labels changed when a concept shift occurs.
        env_batch_size: int
        unique_samples_per_dataset: int
        """
        super().__init__(
            concept_duration=concept_duration,
            num_concept_shifts=num_concept_shifts,
            num_labels_changed=num_labels_changed,
            env_batch_size=env_batch_size,
            unique_samples_per_dataset=unique_samples_per_dataset,
            seed=seed,
            device=device
        )

        self.obs_dim = (784,)
        self.act_dim = 10

        # Load and shuffle data.
        self.images, self.labels, _, _ = load_mnist_data(rng=self.env_rng)
        
        self.images = self.images[:self.unique_samples_per_dataset]
        self.labels = self.labels[:self.unique_samples_per_dataset]
        
        self.images.to(device)
        self.labels.to(device)
        
        self.images = torch.reshape(self.images, (len(self.images), -1))

        all_indices = np.arange(len(self.labels))
        self.env_rng.shuffle(all_indices)
        self.labels = self.labels[all_indices]

        self.task_images = self.images[:]
        self.task_labels = self.labels[:]
        self.task_id_to_new_labels[0] = self.labels[:]
    
    
class RandomLabelCIFAR(RandomLabelEnv):
    def __init__(
        self, 
        concept_duration,
        num_concept_shifts,
        num_labels_changed=10,
        env_batch_size=16, 
        unique_samples_per_dataset=10000,
        seed=0,
        device=None):
        """
        concept_duration: int, number of time steps after which concept shifts.
        num_concept_shifts: int, number of times that the concept shifts.
        number_of_labels_changed: int, number of labels changed when a concept shift occurs.
        env_batch_size: int
        unique_samples_per_dataset: int
        """
        super().__init__(
            concept_duration=concept_duration,
            num_concept_shifts=num_concept_shifts,
            num_labels_changed=num_labels_changed,
            env_batch_size=env_batch_size,
            unique_samples_per_dataset=unique_samples_per_dataset,
            seed=seed,
            device=device
        )
        self.obs_dim = (3, 32, 32)
        self.act_dim = 10
        
        # Load and shuffle data.
        self.images, self.labels, _, _ = load_cifar_data(rng=self.env_rng)
        
        self.images = self.images[:self.unique_samples_per_dataset]
        self.labels = self.labels[:self.unique_samples_per_dataset]
        
        self.images.to(device)
        self.labels.to(device)

        all_indices = np.arange(len(self.labels))
        self.env_rng.shuffle(all_indices)
        self.labels = self.labels[all_indices]

        self.task_images = self.images[:]
        self.task_labels = self.labels[:]
        self.task_id_to_new_labels[0] = self.labels[:]
