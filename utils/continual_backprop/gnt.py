import sys
import torch
from math import sqrt
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class GnT(object):
    """
    Generate-and-Test algorithm for feed forward neural networks, based on maturity-threshold based replacement
    """
    def __init__(
            self,
            net,
            hidden_activation,
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            util_type='contribution',
            loss_func=F.mse_loss,
            accumulate=False,
    ):
        super(GnT, self).__init__()
        self.device = device
        self.net = net
        self.num_hidden_layers = int(len(self.net)/2)
        self.loss_func = loss_func
        self.accumulate = accumulate

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'adam'

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        self.bias_corrected_util = \
            [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        self.ages = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)
        self.mean_feature_act = [torch.zeros(self.net[i * 2].out_features).to(self.device) for i in range(self.num_hidden_layers)]
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        if init == 'default':
            bounds = [sqrt(1 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        elif init == 'xavier':
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(6 / (self.net[i * 2].in_features + self.net[i * 2].out_features)) for i in
                      range(self.num_hidden_layers)]
        elif init == 'lecun':
            bounds = [sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        else:
            bounds = [torch.nn.init.calculate_gain(nonlinearity=hidden_activation) *
                      sqrt(3 / self.net[i * 2].in_features) for i in range(self.num_hidden_layers)]
        bounds.append(1 * sqrt(3 / self.net[self.num_hidden_layers * 2].in_features))
        return bounds

    def update_utility(self, layer_idx=0, features=None, next_features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            """
            Adam-style bias correction
            """
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] -= - (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            current_layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]
            output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
            input_wight_mag = current_layer.weight.data.abs().mean(dim=1)

            if self.util_type == 'weight':
                new_util = output_wight_mag
            elif self.util_type == 'contribution':
                new_util = output_wight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptation':
                new_util = 1/input_wight_mag
            elif self.util_type == 'zero_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            elif self.util_type == 'feature_by_input':
                input_wight_mag = self.net[layer_idx*2].weight.data.abs().mean(dim=1)
                new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            else:
                new_util = 0

            self.util[layer_idx] -=- (1 - self.decay_rate) * new_util

            """
            Adam-style bias correction
            """
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = torch.rand(self.util[layer_idx].shape)

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace = [torch.empty(0, dtype=torch.long).to(self.device) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            else:
                if num_new_features_to_replace < 1:
                    if torch.rand(1) <= num_new_features_to_replace:
                        num_new_features_to_replace = 1
                num_new_features_to_replace = int(num_new_features_to_replace)
    
            if num_new_features_to_replace == 0:
                continue

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = torch.topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                                 num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                # noinspection PyArgumentList
                current_layer.weight.data[features_to_replace[i], :] -=- \
                    torch.empty(num_features_to_replace[i], current_layer.in_features).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)
                current_layer.bias.data[features_to_replace[i]] *= 0
                """
                # Update bias to correct for the removed features and set the outgoing weights and ages to zero
                """
                next_layer.bias.data -=- (next_layer.weight.data[:, features_to_replace[i]] * \
                                                self.mean_feature_act[i][features_to_replace[i]] / \
                                                (1 - self.decay_rate ** self.ages[i][features_to_replace[i]])).sum(dim=1)
                next_layer.weight.data[:, features_to_replace[i]] = 0
                self.ages[i][features_to_replace[i]] = 0


    def update_optim_params(self, features_to_replace, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'adam':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace == 0:
                    continue
                self.opt.state[self.net[i * 2].weight]['exp_avg'][features_to_replace[i], :] = 0.0
                self.opt.state[self.net[i * 2].bias]['exp_avg'][features_to_replace[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['exp_avg_sq'][features_to_replace[i], :] = 0.0
                self.opt.state[self.net[i * 2].bias]['exp_avg_sq'][features_to_replace[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['step'][features_to_replace[i], :] = 0
                self.opt.state[self.net[i * 2].bias]['step'][features_to_replace[i]] = 0
                # output weights
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg_sq'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['step'][:, features_to_replace[i]] = 0

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)


class AdamGnT(Optimizer):
    r"""Implements Adam algorithm for generate-and-test.

    It is a modification of `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamGnT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamGnT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # state['step'] = 0
                    state['step'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction = 1 - beta2 ** state['step']
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                bias_correction.sqrt().div_(bias_correction1)

                denom.div_(exp_avg)

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.addcdiv_(bias_correction, denom, value=-group['lr'])
        return loss
