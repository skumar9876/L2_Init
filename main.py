import wandb
import numpy as np
import torch
import torch.nn.functional as F
import random
import hydra

from omegaconf import OmegaConf

import environments
import agents
import nets
import utils
from tqdm import trange


def compute_acc(env, out_xs, out_ys, agent):
    with torch.no_grad():
        batch_x = out_xs
        avg_acc = 0.
        
        done = False
        index = 0
        
        while not done:
            start = index * env.env_batch_size
            end = (index + 1) * env.env_batch_size

            preds = agent.predict(batch_x[start:end])
            discrete_preds = torch.argmax(preds.reshape(env.env_batch_size, -1), dim=-1)
            acc = discrete_preds == out_ys[start:end].reshape(-1)
            acc = torch.mean(acc.float())
            
            avg_acc += acc
            
            if (index + 1) * env.env_batch_size >= len(batch_x) or (index + 2) * env.env_batch_size > len(batch_x):
                done = True

            index += 1
            
        # batch_size = batch_x.shape[0]
        avg_acc /= float(index)

    metrics = {'acc': avg_acc}
    return metrics


def evaluate(env, agent, train=True, task_id=0):
    out_xs, out_ys = env.get_all_task_data(task_id, train=train)
    metrics = compute_acc(env, out_xs, out_ys, agent) 
    return metrics

def evaluate_on_past_tasks(env, agent, past_task_xs, past_task_ys):
    all_task_metrics = {}

    for task_id in range(len(past_task_xs)):
        out_xs = past_task_xs[task_id]
        out_ys = past_task_ys[task_id]

        metrics = compute_acc(env, out_xs, out_ys, agent)

        all_task_metrics[f'task_{task_id}_acc'] = metrics['acc']

    return all_task_metrics

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    # Seeds
    random.seed(cfg.main.seed)
    np.random.seed(cfg.main.seed)
    torch.manual_seed(cfg.main.seed)
    torch.cuda.manual_seed(cfg.main.seed)
 
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() and cfg.main.device == "gpu" else "cpu")
    print(f"Device: {device}")

    # Logger
    logging_cfg = OmegaConf.to_container(cfg, resolve=True)
    if cfg.main.use_wandb:
        wandb.init(**cfg.wandb, config=logging_cfg)

    # Environment
    env_constructor = environments.__dict__[cfg.env.name]
    env = env_constructor(**cfg.env.params, seed=cfg.main.seed, device=device)

    # Model Constructor
    model_constructor = lambda apply_layer_norm=False, use_crelu=False, fraction_to_remove=0.0: nets.__dict__[cfg.model.model_name](
        input_size=env.obs_dim[0], hidden_size=cfg.model.hidden_size, 
        num_channels=cfg.model.num_channels,
        num_hidden=cfg.model.num_hidden, num_classes=cfg.model.output_dim,
        init_type=cfg.model.init_type,
        apply_layer_norm=apply_layer_norm, 
        use_crelu=use_crelu, fraction_to_remove=fraction_to_remove)
    
    # Loss Function
    loss_fn = utils.__dict__[cfg.main.loss_fn]

    # Agent
    agent_constructor = agents.__dict__[cfg.agent.name]

    if cfg.agent.params is None:
        agent = agent_constructor(model_constructor,
                                  optimizer_cfg=cfg.optimizer_cfg,
                                  loss_fn=loss_fn,
                                  device=device)
    else:
        agent = agent_constructor(model_constructor,
                                  optimizer_cfg=cfg.optimizer_cfg,
                                  device=device,
                                  loss_fn=loss_fn,
                                  **cfg.agent.params)
    
    curr_task_ids = np.zeros(env.horizon // cfg.logging.log_freq)
    curr_task_accs = np.zeros(env.horizon // cfg.logging.log_freq)

    avg_acc = 0.
    curr_task_avg_acc = 0.

    # Used if measuring performance on past tasks to evaluate catastrophic forgetting.
    all_past_task_xs = []
    all_past_task_ys = []

    # Used to store task test accuracies after training on that task.
    task_test_accs = []

    # One Step BWT metric.
    one_step_bwt_metric = 0.

    for i in trange(env.horizon, desc="env steps"):
        
        # Take an environment step.
        x, y, task_id, task_done, curr_task_timestep = env.get_next_sample()
        
        # Take an agent step.
        logits, step_metrics = agent.step(x, y)

        # Compute and log performance metrics (acc, avg_acc)
        with torch.no_grad():
            y_hat_discrete = logits.squeeze().argmax(dim=-1)
            acc = torch.mean((y == y_hat_discrete).float()).detach()
        avg_acc = (i * avg_acc + acc) / (i + 1)
        
        if i % cfg.logging.log_freq == 0:
            metrics = {'acc': acc,
                       'avg_acc': avg_acc,
                       'iter': i,
                       'task_id': task_id}
            if cfg.main.use_wandb:
                wandb.log(metrics)
        
        # Update average accuracy for the current task.
        curr_task_avg_acc = (curr_task_timestep * curr_task_avg_acc + acc) / (curr_task_timestep + 1)

        task_fraction = float(curr_task_timestep) / env.current_task_length
        # Log current task metrics
        if task_fraction in [0.1, 0.25, 0.5, 0.75] or task_done:

            if task_done: task_fraction = 1

            # Eval on test data.
            eval_test_metrics = evaluate(env, agent, train=False, task_id=task_id)

            iter_metrics = {f'curr_task_id_fraction={task_fraction}': task_id,
                            f'curr_task_online_avg_acc_fraction={task_fraction}': curr_task_avg_acc,
                            f'curr_task_test_final_acc_fraction={task_fraction}': eval_test_metrics['acc']
                            }
            
            iter_metrics_task_type = {
                f'task_type={env.task_type}/curr_task_id_fraction={task_fraction}': env.task_type_ids[env.task_type],
                f'task_type={env.task_type}/curr_task_online_avg_acc_fraction={task_fraction}': curr_task_avg_acc,
                f'task_type={env.task_type}/curr_task_test_final_acc_fraction={task_fraction}': eval_test_metrics['acc']
                }
                
            past_task_acc_metrics = {}

            # If task is completed, record test accuracy.
            if task_done: task_test_accs.append(eval_test_metrics['acc'])

            # If task is completed and task_id > 0, evaluate on previous tasks 
            # and add accuracy to one_step bwt.
            if task_done and task_id > 0:
                acc_on_previous_task = evaluate(env, agent, train=False, task_id=task_id - 1)['acc']
                one_step_bwt_metric += acc_on_previous_task - task_test_accs[task_id - 1]

            if task_done:
                if cfg.agent.name in ['EWCAgent', 'L2InitPlusEWCAgent']:

                    # Get completed task's test data.
                    task_test_xs, task_test_ys = env.get_all_task_data(task_id, train=False)

                    # Shuffle data.
                    indices = np.arange(len(task_test_xs))
                    np.random.shuffle(indices)
                    task_test_xs = task_test_xs[indices][:1000]
                    task_test_ys = task_test_ys[indices][:1000]

                    # Update Fisher matrix using this data.
                    agent.update_params_and_fisher(
                        task_test_xs, task_test_ys, batch_size=env.env_batch_size)


                next_task_id = task_id + 1
                # Get data from the next task (which will be used to compute metrics)
                next_task_test_xs, _ = env.get_all_task_data(task_id=next_task_id, train=False)
                # Compute metrics (including number of dead neurons, feature rank, weight magnitude, etc).
                activation_statistics = agent.compute_activation_statistics(next_task_test_xs)
                iter_metrics.update(activation_statistics)
                
                for key in activation_statistics:
                    iter_metrics_task_type[f'task_type={env.task_type}_{key}'] = activation_statistics[key]
                
                curr_task_avg_acc = 0.


                past_task_acc_metrics = {}
                if cfg.main.measure_forgetting:
                    # If measuring performance on past tasks, collect data from completed task.
                    sample_task_xs, sample_task_ys = env.get_all_task_data(task_id=task_id, train=False)
                    
                    # Get 1000 random samples.
                    dataset_size = sample_task_xs.shape[0]
                    num_samples = 1000
                    random_indices = np.random.choice(
                        dataset_size, size=num_samples, replace=False)

                    # Select samples using random indices.
                    selected_xs = sample_task_xs[random_indices]
                    selected_ys = sample_task_ys[random_indices]

                    # Append to past task data.
                    all_past_task_xs.append(selected_xs)
                    all_past_task_ys.append(selected_ys)
                    
                    # Evaluate model on past tasks.
                    past_task_acc_metrics = evaluate_on_past_tasks(
                        env, agent, all_past_task_xs, all_past_task_ys)

            if cfg.main.use_wandb:
                wandb.log({**iter_metrics, **iter_metrics_task_type, **past_task_acc_metrics})

    # Compute backward transfer metric as computed in https://arxiv.org/pdf/1706.08840.pdf.
    if cfg.main.measure_forgetting:
        bwt_metric = 0.
        for i in range(task_id):
            # Compute accuracy on past task.
            new_past_task_acc = evaluate(env, agent, train=False, task_id=i)['acc']
            bwt_metric += new_past_task_acc - task_test_accs[i]
        
        if cfg.main.use_wandb:
            bwt_metric /= task_id
            one_step_bwt_metric /= task_id
            wandb.log({"bwt": bwt_metric, "one_step_bwt": one_step_bwt_metric, 'overall_avg_acc': avg_acc})


    wandb.finish()

if __name__ == '__main__':
    main()
