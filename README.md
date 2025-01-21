# Maintaining Plasticity in Continual Learning via Regenerative Regularization

## Overview
This repository contains the code to reproduce the experiments present in our paper titled [Maintaining Plasticity in Continual Learning via Regenerative Regularization](https://arxiv.org/pdf/2308.11958). A talk about this work can be found [here](https://www.youtube.com/watch?v=aovFA2Or0Ok).

## Abstract
In continual learning, plasticity refers to the ability of an agent to quickly adapt to new information. Neural networks are known to lose plasticity when processing non-stationary data streams. In this paper, we propose L2 Init, a simple approach for maintaining plasticity by incorporating in the loss function L2 regularization toward initial parameters. This is very similar to standard L2 regularization (L2), the only difference being that L2 regularizes toward the origin. L2 Init is simple to implement and requires selecting only a single hyper-parameter. The motivation for this method is the same as that of methods that reset neurons or parameter values. Intuitively, when recent losses are insensitive to particular parameters, these parameters should drift toward their initial values. This prepares parameters to adapt quickly to new tasks. On problems representative of different types of nonstationarity in continual supervised learning, we demonstrate that L2 Init most consistently mitigates plasticity loss compared to previously proposed approaches.

## Repository contents:
- [agents](https://github.com/skumar9876/L2_Init/tree/main/agents): All the algorithms used in the paper, including our L2 Init regularization.
- [nets](https://github.com/skumar9876/L2_Init/tree/main/nets): The network architectures used in the paper.
- [environments](https://github.com/skumar9876/L2_Init/tree/main/environments): The environments used in the paper: Permuted MNIST, Random Label MNIST, Random Label CIFAR, and 5+1 CIFAR.

## Setup
Clone the repo:
``` sh
git clone https://github.com/skumar9876/L2_Init.git
cd L2_Init
```

Set up a conda environment and install packages:
``` sh
conda create -n regen_reg python=3.10 --yes
conda activate regen_reg
pip install -r requirements.txt
```

## Train agents.
The training scripts are located in [utils/commands/train](https://github.com/skumar9876/L2_Init/tree/main/utils/commands/train). There is one script for each environment. To train all algorithms on Permuted MNIST, run the following command:
``` sh
python utils/commands/train/parallel_runs_permuted_mnist.py
```