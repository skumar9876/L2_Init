from joblib import Parallel, delayed
from itertools import product
import subprocess
import multiprocessing

def run_command(command):
    """
    Function to run a command.
    """
    subprocess.run(command, shell=True)
    
def generate_commands(base_command, params):
    """
    Generate a list of commands based on the base command and parameters.
    """
    keys = list(params.keys())
    values = list(params.values())
    # Generate all combinations of parameters
    combinations = list(product(*values))
    commands = []
    for combination in combinations:
        command = base_command
        for key, value in zip(keys, combination):
            command += f" {key}={value}"
        commands.append(command)
    return commands

def create_all_commands():
    
    SWEEP='sweep=TRAIN'
    
    ENV_ARGS="env=random_label_mnist env.params.unique_samples_per_dataset=1200 env.params.concept_duration=30000 \
          env.params.num_concept_shifts=50 env.params.env_batch_size=16"

    MODEL_ARGS="model.model_name=MLP model.output_dim=10 model.num_hidden=2 model.hidden_size=100"

    EXP_ARGS="logging.log_freq=13037 main.use_wandb=True"
    
    base_command_str = f'python main.py optimizer_cfg.name=SGD {ENV_ARGS} {MODEL_ARGS} {EXP_ARGS} {SWEEP}'
    lr_params = {'optimizer_cfg.lr': [str(0.01), str(0.001)]}
    
    # base_command_str = f'python main.py optimizer_cfg.name=Adam {ENV_ARGS} {MODEL_ARGS} {EXP_ARGS} {SWEEP}'
    # lr_params = {'optimizer_cfg.lr': [str(0.001), str(0.0001)]}
    
    # Batch 1 - BaseAgent.
    command_str = f'{base_command_str} agent=base'
    params = {
        'main.seed': [str(i) for i in range(3)],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch1 = generate_commands(command_str, params)

    
    # Batch 2 - L2InitAgent.
    command_str = f'{base_command_str} agent=l2_init'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.l2_weight': [str(0.01), str(0.001), str(0.0001), str(0.00001)],
        'agent.params.sample_init_values': [False],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch2 = generate_commands(command_str, params)

    # Batch 3 - L2Agent.
    command_str = f'{base_command_str} agent=l2'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.l2_weight': [str(0.01), str(0.001), str(0.0001), str(0.00001)],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch3 = generate_commands(command_str, params)
    
    # Batch 4 - L1InitAgent.
    command_str = f'{base_command_str} agent=l1_init'
    params = {
        'main.seed': [str(i) for i in range(3, 13)],
        'agent.params.l1_weight': [str(0.001)],  # [str(0.01), str(0.001), str(0.0001), str(0.00001)]
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch4 = generate_commands(command_str, params)
    
    # Batch 5 - CReLU.
    command_str = f'{base_command_str} agent=crelu'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.fraction_to_remove': [str(0.09)],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch5 = generate_commands(command_str, params)
    
    # Batch 6 - ContinualBackpropAgent
    command_str = f'{base_command_str} \
                    agent=continual_backprop \
                    agent.params.decay_rate=0.99 \
                    agent.params.maturity_threshold=100 \
                    agent.params.util_type=adaptable_contribution \
                    agent.params.accumulate=False'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.replacement_rate': [str(0.0001), str(0.00001), str(0.000001), str(0.1), str(0.01), str(0.001)], 
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch6 = generate_commands(command_str, params)
    
    # Batch 7 - Layer Norm
    command_str = f'{base_command_str} agent=layer_norm'
    params = {
        'main.seed': [str(i) for i in range(3)],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch7 = generate_commands(command_str, params)
    
    # Batch 8 - ReDO
    command_str = f'{base_command_str} agent=redo'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.recycle_period': [780],
        'agent.params.recycle_threshold': [0.0, 0.01, 0.1],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch8 = generate_commands(command_str, params)
    
    # Batch 9 - Shrink and Perturb
    command_str = f'{base_command_str} agent=shrink_and_perturb'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.shrink': [1e-4],
        'agent.params.perturb_scale': [1e-2],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch9 = generate_commands(command_str, params)
    
    # Batch 10 - L2 Init + Resample.
    command_str = f'{base_command_str} agent=l2_init'
    params = {
        'main.seed': [str(i) for i in range(3)],
        'agent.params.l2_weight': [str(0.01)],  # [str(0.01), str(0.001), str(0.0001), str(0.00001)]
        'agent.params.sample_init_values': [True],
    }
    params = {
        **params,
        **lr_params,
    }
    commands_batch10 = generate_commands(command_str, params)
    

    commands = commands_batch6
    
    return commands

if __name__ == "__main__":
    
   
    commands = create_all_commands()
    #print("Commands", commands)
    print("Num commands", len(commands))
    # Using joblib to parallelize the execution
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(run_command)(command) for command in commands)