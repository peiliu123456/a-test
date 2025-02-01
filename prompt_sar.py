from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import timm
import json
from utils.utils import get_imagenet_r_mask
import pandas as pd

imagenet_r_mask = get_imagenet_r_mask()


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class Prompt_sar(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """

    def __init__(self, args, model, optimizer, steps=1, episodic=False, margin_e0=0.4 * math.log(1000),
                 reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SAR requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.idx = 0

        self.margin_e0 = margin_e0  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = reset_constant_em  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.args = args

    def forward(self, x, y):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt_sar(x, y, self.model, self.optimizer, self.margin_e0,
                                                                  self.reset_constant_em, self.ema, self.args)
            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def cal_grad(self, item):
        total_grad = 0.0
        total_param = 0
        # 遍历模型的所有参数
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):  # 判断该层是否为 LayerNorm 层
                for param_name, param in module.named_parameters():
                    if param.grad is not None:
                        # 计算该层参数的梯度的均值，并累加
                        total_grad += param.grad.abs().mean().item()  # 使用绝对值的均值
                        total_param += 1  # 计算 LN 层参数的数量

        # 创建或读取 Excel 文件，假设文件名为 "data.xlsx"
        total_grad2 = 0.0
        total_param2 = 0
        # 遍历模型的所有参数
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 计算每个参数的梯度的均值，并累加
                total_grad2 += param.grad.abs().mean().item()  # 使用绝对值的均值
                total_param2 += 1  # 计算有梯度的参数数量
        avg_grad2 = total_grad2 / total_param2
        avg_grad = total_grad / total_param
        grad = abs(avg_grad - avg_grad2)
        file_path = 'grad.xlsx'

        # 如果 Excel 文件存在，读取现有数据
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            # 如果文件不存在，创建一个空的 DataFrame
            df = pd.DataFrame(columns=["x"])

        if total_param > 0:
            avg_grad = total_grad / total_param
            if not np.isnan(item):
                # 创建一个新的 DataFrame 行
                new_row = pd.DataFrame([{"x": avg_grad}])
                # 使用 pd.concat 将新行追加到原 DataFrame
                df = pd.concat([df, new_row], ignore_index=True)
                # 将更新后的 DataFrame 写回到 Excel 文件
                df.to_excel(file_path, index=False)
                # 打印更新后的 DataFrame 以验证
        else:
            print("模型没有梯度")

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sar(self, x, y, model, optimizer, margin, reset_constant, ema, args):
        """Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        outputs = model(x)

        reset_flag = False
        #loss = softmax_entropy(outputs).mean(0)
        ce = nn.CrossEntropyLoss()

        loss = ce(outputs, y)
        loss.backward()
        # 使用 argmax 找到每个样本的预测类别
        if self.idx < 500:
            flag=1
            self.cal_grad(flag)
        self.idx = self.idx + 1
        if self.idx % 50 == 0:
            print(loss.item())
        if self.idx>=500:
            print("done!!!!!!!!!!!!!!!!")
        optimizer.step()
        optimizer.zero_grad()
        if not np.isnan(loss.item()):
            self.ema = update_ema(self.ema,
                                  loss.item())  # record moving average loss values for model recovery
        if self.ema is not None:
            if self.ema < 0.2:
                # print("ema < 0.2, now reset the model")
                reset_flag = True
        return outputs, self.ema, False


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model, args):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'blocks.21' in nm:
            continue
        if 'blocks.22' in nm:
            continue
        if 'blocks.23' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    # params.append(p)
                    p.requires_grad_(True)
                    params += [{'params': p, 'lr': args.lr}]
                    names.append(f"{nm}.{np}")
        for np, p in m.named_parameters():
            if np in ['prompt_emb']:  # weight is scale, bias is shift
                # params.append(p)
                p.requires_grad_(True)
                params += [{'params': p, 'lr': args.lr}]
                names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
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
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"
