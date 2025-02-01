from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
from pycm import *
import math
from typing import ValuesView
from utils.utils import get_logger, get_imagenet_r_mask
from dataset.selectedRotateImageFolder import prepare_test_data, prepare_random_oneshot_test_data
from utils.cli_utils import *
import torch
import torch.nn.functional as F
import tent, sar, dct, dpal, ours, prompt_sar, cotta
from sam import SAM
from gnp import GNP
import timm
import models.Res as Resnet
from models.dct_attention import DCT_Attention
from models.dpal_transformer import DPAL_Transformer
from models.ours_transformer import OURS_Transformer
from models.Prompt_sar_transformer import Prompt_sar_Transformer
import copy, json
import clip
from robustbench.data import load_cifar10c
from robustbench.data import load_cifar100c
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
import torchvision.datasets
from copy import deepcopy
import cv2
import numpy as np
import torch
import timm



def validate(val_loader, model, args, repeat, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    end = time.time()
    for i, dl in enumerate(val_loader):
        images, target = dl[0].cuda(), dl[1].cuda()
        if repeat:
            with torch.no_grad():
                if args.method in ['dpal', 'ours']:
                    output, _ = model.forward_repeat(images)
                else:
                    output = model.forward_repeat(images)
        else:
            output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
        if i > 10 and args.debug:
            break
    return top1.avg, top5.avg


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


@torch.no_grad()
def load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_prefix_tokens', 1),
            model.patch_embed.grid_size
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))

    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def get_args():
    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--dataset', default='imagenet-c', help='path to dataset')
    parser.add_argument('--data_corruption', default='E:\imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar')
    parser.add_argument('--model', default='vitbase_timm', type=str,
                        help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='normal', type=str, help='normal, mix_shifts, bs1, label_shifts')

    parser.add_argument('--sar_margin_e0', default=math.log(1000) * 0.40, type=float,
                        help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float,
                        help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).')
    parser.add_argument('--LN_lr', default=0.05, type=float, help='LN_lr')
    parser.add_argument('--dct_lr', default=0.01, type=float, help='dct_lr')
    parser.add_argument('--lr', default=1e-3, type=float, help='lr')
    parser.add_argument('--prompt_lr', default=1e-2, type=float, help='prompt_lr')
    parser.add_argument('--predictor_lr', default=1e-2, type=float, help='predictor_lr')
    parser.add_argument('--prompt_deep', type=int, default=1, help='prompt_deep')
    parser.add_argument('--GNP', action='store_true', help='use GNP')
    parser.add_argument('--dual_prompt_tokens', default=2, type=int, help='dual_prompt_tokens')
    parser.add_argument('--num_prompt_tokens', default=1, type=int, help='num_prompt_tokens')
    parser.add_argument('--img_size', default=224, type=int, help='vit_image_size')
    parser.add_argument('--num_classes', default=1000, type=int, help='vit_head')
    parser.add_argument('--repeat', default=False, type=bool, help='use to repeat')
    parser.add_argument('--alpha', default=0.8, type=float, help='use to repeat')
    parser.add_argument('--rho', default=0.01, type=float, help='use to repeat')
    parser.add_argument('--TTA', action='store_true', help='enable TTA')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    torch.cuda.set_device(args.device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    model_path = 'vit_base_patch16_224.pth'
    if not os.path.exists(model_path):
        print(f"{model_path} dont exit, need to save")
        net = timm.create_model('vit_base_patch16_224', pretrained=True)
        torch.save(net.state_dict(), 'vit_base_patch16_224.pth')
        print(f"model save to {model_path}")
    else:
        print(f"model exit!")
    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    print(args.output)
    if not os.path.exists(args.output):  # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)
    args.logger_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-{}-{}-level{}-seed{}.txt".format(
        args.method, args.model, args.level, args.seed)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False)
    #
    if args.dataset == 'imagenet-c':
        common_corruptions = common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                                   'glass_blur',
                                                   'motion_blur',
                                                   'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                                                   'elastic_transform',
                                                   'pixelate', 'jpeg_compression']
    elif args.dataset == 'VisDA-2021':
        common_corruptions = ['VisDA-2021']
    elif args.dataset == 'imagenet-r':
        common_corruptions = ['imagenet-r']
    elif args.dataset == 'imagenet-a':
        common_corruptions = ['imagenet-a']
    acc1s, acc5s = [], []
    fewshot_acc1s = []
    ir = args.imbalance_ratio
    bs = args.test_batch_size
    if args.method in ['tent', 'sar', 'no_adapt', 'dct', 'dpal', 'ours', 'prompt_sar', 'cotta']:
        if args.model == "vitbase_timm":
            if args.method == 'dct':
                net = timm.create_model('vit_base_patch16_224', pretrained=False)
                for block in net.blocks[:9]:
                    block.attn = DCT_Attention().cuda()
                # load_weights(net, checkpoint_path='./models/vit_base_patch16_224.npz')
                state_dict = torch.load('vit_base_patch16_224.pth')
                net.load_state_dict(state_dict, strict=False)
            elif args.method == 'dpal':
                net = timm.create_model('vit_base_patch16_224', pretrained=False)
                net = DPAL_Transformer(args)
                # load_weights(net, checkpoint_path='./models/vit_base_patch16_224.npz')
                state_dict = torch.load('vit_base_patch16_224.pth')
                net.load_state_dict(state_dict, strict=False)
            elif args.method == 'ours':
                net = OURS_Transformer(args)
                state_dict = torch.load('vit_base_patch16_224.pth')
                net.load_state_dict(state_dict, strict=False)
            elif args.method == 'prompt_sar':
                net = Prompt_sar_Transformer(args)
                state_dict = torch.load('vit_base_patch16_224.pth')
                net.load_state_dict(state_dict, strict=False)
            elif args.method in ['sar', 'no_adapt', 'tent', 'cotta']:
                net = timm.create_model('vit_base_patch16_224', pretrained=False)
                state_dict = torch.load('vit_base_patch16_224.pth')
                net.load_state_dict(state_dict, strict=False)
        net = net.cuda()
    else:
        assert False, NotImplementedError
    if args.method == "tent":
        net = tent.configure_model(net)
        params, param_names = tent.collect_params(net)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_net = tent.Tent(net, optimizer, args)

    elif args.method == "cotta":
        net = cotta.configure_model(net)
        params, param_names = cotta.collect_params(net)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_net = cotta.CoTTA(net, optimizer)
    elif args.method == "no_adapt":
        adapt_net = net
        net.eval()
    elif args.method in ['sar']:
        net = sar.configure_model(net)
        params, param_names = sar.collect_params(net, args)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, momentum=0.9)
        adapt_net = sar.SAR(args, net, optimizer, margin_e0=args.sar_margin_e0)
    elif args.method == 'dct':
        net = dct.configure_model(net)
        params, param_names = dct.collect_params(net, args)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, momentum=0.9)
        adapt_net = dct.DCT(args, net, optimizer, margin_e0=args.sar_margin_e0)
    elif args.method == 'dpal':
        net = dpal.configure_model(net)
        params, param_names = dpal.collect_params(net, args)
        logger.info(param_names)
        prompt_params, prompt_param_names = dpal.collect_prodictor_params(net, args)
        logger.info(prompt_param_names)
        optimizer = SAM(params, torch.optim.SGD, momentum=0.9)
        ad_optimizer = torch.optim.SGD(prompt_params, momentum=0.9)
        adapt_net = dpal.DPAL(net, args, optimizer, ad_optimizer, margin_e0=args.sar_margin_e0)
    elif args.method == 'ours':
        if args.GNP == False:
            net = ours.configure_model(net)
            params, param_names = ours.collect_params(net, args)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, momentum=0.9)
            adapt_net = ours.OURS(net, args, optimizer, margin_e0=args.sar_margin_e0)
        else:
            net = ours.configure_model(net)
            params, param_names = ours.collect_params(net, args)
            logger.info(param_names)
            optimizer = GNP(params, torch.optim.SGD, momentum=0.9, rho=args.rho, alpha=args.alpha)
            adapt_net = ours.OURS(net, args, optimizer, margin_e0=args.sar_margin_e0)
    elif args.method == 'prompt_sar':
        net = prompt_sar.configure_model(net)
        params, param_names = prompt_sar.collect_params(net, args)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        optimizer = torch.optim.SGD(params, momentum=0.9)
        adapt_net = prompt_sar.Prompt_sar(args, net, optimizer, margin_e0=args.sar_margin_e0)
    else:
        assert False, NotImplementedError
    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")
    for corrupt in common_corruptions:
        if args.TTA:
            adapt_net.reset()
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs
        if args.method in ['tent', 'sar', 'no_adapt', 'dct', 'ours', 'dpal', 'prompt_sar', 'cotta']:
            if args.corruption != 'mix_shifts':
                if args.dataset == 'VisDA-2021':
                    from utils.get_visda2021 import get_dataloaders

                    args.evaluation_data = './datasets/VisDA-2021/val_filelists/imagenet_c_r.txt'
                    val_loader = get_dataloaders(args)
                    val_loader.dataset.labels[val_loader.dataset.labels > 1000] = 1000
                elif args.dataset == 'imagenet-r':
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    test_transform = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                         torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
                    imagenet_r = torchvision.datasets.ImageFolder(root="./imagenet-r", transform=test_transform)
                    val_loader = torch.utils.data.DataLoader(imagenet_r, batch_size=args.test_batch_size,
                                                             shuffle=False,
                                                             num_workers=0, pin_memory=True)
                elif args.dataset == 'imagenet-a':
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    test_transform = torchvision.transforms.Compose(
                        [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224),
                         torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
                    imagenet_a = torchvision.datasets.ImageFolder(root="./imagenet-a", transform=test_transform)
                    val_loader = torch.utils.data.DataLoader(imagenet_a, batch_size=args.test_batch_size,
                                                             shuffle=False,
                                                             num_workers=0, pin_memory=True)
                elif args.dataset == 'imagenet-c':
                    val_dataset, val_loader = prepare_test_data(args)
                    val_dataset.switch_mode(True, False)
        else:
            assert False, NotImplementedError
        if args.exp_type == 'label_shifts':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2021:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed,
                                                                                                    100000, ir)
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())
        logger.info(args)
        top1, top5 = validate(val_loader, adapt_net, args, mode='eval', repeat=False)
        logger.info(
            f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1:.5f} and top5: {top5:.5f}")
        acc1s.append(top1.item())
        avg1 = sum(acc1s) / len(acc1s)
        logger.info(f"Average acc1s are {avg1}")
        acc5s.append(top5.item())
        logger.info(f"acc1s are {acc1s}")
        logger.info(f"acc5s are {acc5s}")
        if args.repeat:
            repeat_model = deepcopy(adapt_net)
            repeat_model.eval()
            acc1s_repeat = []
            logger.info(f"Now, start to repeat!")
            for repeat_c in common_corruptions:
                args.corruption = repeat_c
                val_dataset, val_loader = prepare_test_data(args)
                val_dataset.switch_mode(True, False)
                top1, top5 = validate(val_loader, repeat_model, args, mode='eval', repeat=True)
                # logger.info(
                #     f"Result under {repeat_c}. The adaptation accuracy of {args.method} is top1: {top1:.5f} and top5: {top5:.5f}")
                acc1s_repeat.append(top1.item())
                avg1_repeat = sum(acc1s_repeat) / len(acc1s_repeat)
                logger.info(f"Repeat Average are {avg1_repeat}")
                # logger.info(f"Repeat acc1s are {acc1s_repeat}")
                if repeat_c == corrupt:
                    del repeat_model
                    break
