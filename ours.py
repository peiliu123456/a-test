"""
Copyright to DPAL Authors, ECCV 2024
built upon on Tent and SAR code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math, json
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
import torchvision.transforms as transforms
import my_transforms as my_transforms
import PIL

ce = nn.CrossEntropyLoss()


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def update_ema_variables(ema_model, model, alpha_teacher, name_filter=None):
    """
    Update EMA variables for specific parameters (such as 'prompt_emb' and 'predictor').

    Parameters:
    - ema_model: EMA model (target model).
    - model: Model whose parameters are being updated.
    - alpha_teacher: Smoothing coefficient for EMA update.
    - name_filter: A list of parameter names to apply EMA updates to (e.g., ['prompt_emb', 'predictor']).
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # Skip parameters that are not part of 'prompt_emb' or 'predictor' if filter is provided
        if name_filter and not any(name in param.name for name in name_filter):
            continue
        else:
            ema_param.data[:] = alpha_teacher * ema_param.data[:] + (1 - alpha_teacher) * param.data[:]

    return ema_model


def ctfg(list_attentions_a, list_attentions_b, normalize=True, feature_distil_factor=None):
    assert len(list_attentions_a) == len(list_attentions_b)
    loss = torch.tensor(0.).cuda()
    for ii, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, C, embsize)
        assert a.shape == b.shape, (a.shape, b.shape)
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if normalize:
            a = F.normalize(a, dim=2, p=2)
            b = F.normalize(b, dim=2, p=2)
        if feature_distil_factor is None:
            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        else:
            factor = feature_distil_factor[ii].reshape([1, -1])
            layer_loss = torch.mean(factor * torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss
    return loss / len(list_attentions_a)


class OURS(nn.Module):
    """DPAL online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once DPALed, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, args, optimizer, steps=1, episodic=False,
                 margin_e0=0.4 * math.log(1000),
                 reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.old_model0 = None
        self.old_model1 = None
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "DPAL requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.importance = None
        self.alpha = 0.95
        self.freq = 0
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        self.margin_e0 = 0.4 * math.log(args.num_classes)  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mask = None

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.old_model1 = deepcopy(self.model)
        self.old_model0 = deepcopy(self.model)
        self.ema = None

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            self.attention_weights = None
            outputs, reset_flag = self.forward_and_adapt_dual(x)
            if reset_flag:
                self.reset()
        return outputs

    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def getloss(self, outputs):
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            max_probs, labels = torch.max(probs, dim=1)
            mask1 = max_probs >= max_probs.mean(dim=0).item()
            entropys = self.softmax_entropy(outputs)
            mask2 = entropys <= entropys.mean(dim=0).item()
            mask = mask1 & mask2
            if self.args.test_batch_size == 1:
                mask3 = entropys <= self.margin_e0
                mask = mask & mask3
        labels = labels[mask]
        self.mask = mask
        outputs_mask = outputs[mask]
        if not np.isnan(entropys.mean(0).item()):
            self.ema = update_ema(self.ema,
                                  entropys.mean(0).item())  # record moving average loss values for model recovery
        loss = ce(outputs_mask, labels)
        out_p, index = torch.topk(outputs_mask, 3, dim=1)
        labels_p = torch.zeros(index.shape[0], dtype=torch.long).cuda()
        loss_p = ce(out_p / 3., labels_p)
        loss += loss_p
        return loss

    def forward_and_adapt_dual(self, x):
        if self.freq % 2 == 0:
            self.old_model0 = deepcopy(self.model)
        else:
            self.old_model1 = deepcopy(self.model)

        outputs, hidden_layer = self.model(x, self.importance)
        loss = self.getloss(outputs)
        if self.freq != 0:
            with torch.no_grad():
                _, old_hidden_layer = self.old_model1(x,
                                                      self.importance) if self.freq % 2 == 0 else self.old_model0(x,
                                                                                                                  self.importance)
            loss += ctfg(hidden_layer, old_hidden_layer, feature_distil_factor=self.importance)
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        outputs_second, hidden_layer = self.model(x, self.importance)
        loss_second = self.getloss(outputs_second)
        if self.freq != 0: loss_second += ctfg(hidden_layer, old_hidden_layer, feature_distil_factor=self.importance)
        self.freq = self.freq + 1
        loss_second.backward()
        self.normalize_importance()
        self.optimizer.second_step(zero_grad=True)
        reset_flag = False
        if self.ema is not None:
            em = 0.15 if self.args.num_classes == 1000 else 0.01
            if self.ema < em:
                print(f"ema < {em}, now reset the model")
                reset_flag = True

        return outputs, reset_flag

    def forward_repeat(self, x):
        with torch.no_grad():
            return self.model(x, self.importance)

    def normalize_importance(self):
        with torch.no_grad():
            grad = []
            for i, g in enumerate(self.model.block_gradients):
                if g is not None:
                    grad.append(g)
            grad = torch.abs(torch.stack(grad))
            gradients = torch.abs(grad.mean(dim=(1, 3)))
            min_values, _ = gradients.min(dim=1, keepdim=True)
            max_values, _ = gradients.max(dim=1, keepdim=True)
            range_values = max_values - min_values
            range_values[range_values == 0] = 1e-6

            if self.importance is None:
                self.importance = (gradients - min_values) / range_values
            else:
                self.importance = ((gradients - min_values) / range_values) * self.alpha + self.importance * (
                        1 - self.alpha)
            del grad
            self.model.clear_gradients()
        return self.importance


def collect_params(model, args):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    seen_params = set()
    for nm, m in model.named_modules():

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    p.requires_grad_(True)
                    if id(p) not in seen_params:
                        params += [{'params': p, 'lr': args.lr}]
                        seen_params.add(id(p))
                        names.append(f"{nm}.{np}")

        for np, p in m.named_parameters():
            if 'predictor' in np:
                p.requires_grad_(True)
                if id(p) not in seen_params:
                    params += [{'params': p, 'lr': args.prompt_lr}]
                    seen_params.add(id(p))
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())

    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state, ):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Balance Prompt."""
    # train mode, because DPAL optimizes the model to minimize entropy
    model.train()
    model.requires_grad_(False)
    # configure norm for DPAL updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with Balance Prompt."""
    is_training = model.training
    assert is_training, "Balance Prompt needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "Balance Prompt needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "Balance Prompt should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "Balance Prompt needs normalization layer parameters for its optimization"
