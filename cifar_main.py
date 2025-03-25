from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
from pycm import *
import math
from utils.utils import get_logger, get_imagenet_r_mask
from dataset.selectedRotateImageFolder import prepare_test_data, prepare_random_oneshot_test_data
from utils.cli_utils import *
import torch
import torch.nn.functional as F
import tent, sar, dct, dpal, ours, prompt_sar
from sam import SAM
import timm
import models.Res as Resnet
from models.dct_attention import DCT_Attention
from models.dpal_transformer import DPAL_Transformer
from models.ours_transformer import OURS_Transformer
from robustbench.data import load_cifar10c
from robustbench.data import load_cifar100c
from collections import OrderedDict
from copy import deepcopy


def validate(val_loader, model, args, repeat, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        math.ceil(len(val_loader['x']) / args.test_batch_size),
        [batch_time, top1, top5],
        prefix='Test: ')
    end = time.time()
    x = val_loader['x'].cuda()
    y = val_loader['y'].cuda()
    n_batches = math.ceil(x.shape[0] / args.test_batch_size)
    for counter in range(n_batches):
        x_curr = x[counter * args.test_batch_size:(counter + 1) *
                                                  args.test_batch_size]
        y_curr = y[counter * args.test_batch_size:(counter + 1) *
                                                  args.test_batch_size]
        x_curr = torch.nn.functional.interpolate(x_curr, size=(args.img_size, args.img_size),
                                                 mode='bilinear', align_corners=False)
        if repeat:
            with torch.no_grad():
                if args.method in ['dpal', 'ours']:
                    output, _ = model.forward_repeat(x_curr)
                else:
                    output = model.forward_repeat(x_curr)
        else:
            output = model(x_curr)
        acc1, acc5 = accuracy(output, y_curr, topk=(1, 5))
        top1.update(acc1[0], x_curr.size(0))
        top5.update(acc5[0], x_curr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if counter % args.print_freq == 0:
            progress.display(counter)
        if counter > 10 and args.debug:
            break

    return top1.avg, top5.avg


def get_args():
    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--dataset', default='imagenet-c', help='path to dataset')
    parser.add_argument('--data_corruption', default='D:\imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=32, type=int,
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
    parser.add_argument('--alpha', default=0.8, type=float, help='use to GNP')
    parser.add_argument('--rho', default=0.01, type=float, help='use to GNP')

    parser.add_argument('--dual_prompt_tokens', default=2, type=int, help='dual_prompt_tokens')
    parser.add_argument('--num_prompt_tokens', default=1, type=int, help='num_prompt_tokens')
    parser.add_argument('--img_size', default=384, type=int, help='vit_image_size')
    parser.add_argument('--num_classes', default=10, type=int, help='vit_head')
    parser.add_argument('--repeat', default=False, type=bool, help='use to repeat')
    parser.add_argument('--TTA', action='store_true', help='enable TTA')
    return parser.parse_args()


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


if __name__ == '__main__':

    args = get_args()
    torch.cuda.set_device(args.device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    args.output = f"{args.output}/cifar{args.num_classes}c"
    print(args.output)
    if not os.path.exists(args.output):  # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)
    args.logger_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-{}-{}-level{}-seed{}.txt".format(
        args.method, args.model, args.level, args.seed)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False)
    #
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                          'motion_blur',
                          'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
                          'pixelate', 'jpeg_compression']
    acc1s, acc5s = [], []
    fewshot_acc1s = []
    ir = args.imbalance_ratio
    bs = args.test_batch_size
    if args.method in ['tent', 'sar', 'no_adapt', 'dct', 'dpal', 'ours', 'prompt_sar']:
        if args.model == "vitbase_timm":
            if args.method == 'dct':
                net = timm.create_model("vit_base_patch16_384",
                                        pretrained=False, num_classes=args.num_classes)
                for block in net.blocks[:9]:
                    block.attn = DCT_Attention().cuda()
                checkpoint = torch.load(f"vit_base_384_cifar{args.num_classes}.t7",
                                        map_location=torch.device('cpu'))
                checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
                net.load_state_dict(checkpoint, strict=False)
            elif args.method == 'dpal':
                net = DPAL_Transformer(args)
                checkpoint = torch.load(f"vit_base_384_cifar{args.num_classes}.t7",
                                        map_location=torch.device('cpu'))
                checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
                net.load_state_dict(checkpoint, strict=False)
            elif args.method == 'ours':
                net = OURS_Transformer(args)
                checkpoint = torch.load(f"vit_base_384_cifar{args.num_classes}.t7",
                                        map_location=torch.device('cpu'))
                checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
                net.load_state_dict(checkpoint, strict=False)
            elif args.method in ['sar', 'tent', 'no_adapt', 'prompt_sar']:
                net = timm.create_model("vit_base_patch16_384",
                                        pretrained=False, num_classes=args.num_classes)
                checkpoint = torch.load(f"vit_base_384_cifar{args.num_classes}.t7",
                                        map_location=torch.device('cpu'))
                checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
                net.load_state_dict(checkpoint, strict=False)
        net = net.cuda()
    else:
        assert False, NotImplementedError
    if args.method == "tent":
        net = tent.configure_model(net)
        params, param_names = tent.collect_params(net)
        logger.info(param_names)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_net = tent.Tent(net, optimizer, args)
    elif args.method == "no_adapt":
        adapt_net = net
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
        logger.info(param_names)
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
        if args.GNP == True:
            net = ours.configure_model(net)
            params, param_names = ours.collect_params(net, args)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, momentum=0.9)
            adapt_net = ours.OURS(net, args, optimizer, margin_e0=args.sar_margin_e0)
        else:
            net = ours.configure_model(net)
            params, param_names = ours.collect_params(net, args)
            logger.info(param_names)
            optimizer = SAM(params, torch.optim.SGD, momentum=0.9)
            adapt_net = ours.OURS(net, args, optimizer, margin_e0=args.sar_margin_e0)
    elif args.method == 'prompt_sar':
        net = prompt_sar.configure_model(net)
        params, param_names = prompt_sar.collect_params(net, args)
        logger.info(param_names)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, momentum=0.9)
        adapt_net = prompt_sar.Prompt_sar(args, net, optimizer, margin_e0=args.sar_margin_e0)
    else:
        assert False, NotImplementedError

    for corrupt in common_corruptions:
        if args.TTA:
            adapt_net.reset()
        args.corruption = corrupt
        bs = args.test_batch_size
        if args.num_classes in [10, 100]:
            args.print_freq = 10000 // 20 // bs
        if args.method in ['tent', 'sar', 'no_adapt', 'dct', 'ours', 'dpal', 'prompt_sar']:
            if args.corruption != 'mix_shifts':
                load_func = {10: load_cifar10c, 100: load_cifar100c}
                x_test, y_test = load_func[args.num_classes](10000, 5, "./data", False, [corrupt])
                val_loader = {'x': x_test, 'y': y_test}
        else:
            assert False, NotImplementedError
        # construt new dataset with online imbalanced label distribution shifts, see Section 4.3 for details
        # note that this operation does not support mix-domain-shifts exps

        logger.info(args)
        top1, top5 = validate(val_loader, adapt_net, args, mode='eval', repeat=False)
        logger.info(
            f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1:.5f} and top5: {top5:.5f}")
        acc1s.append(top1.item())
        path = f"{args.output}/{args.test_batch_size}_{args.lr}_{args.prompt_lr}_{args.predictor_lr}"
        avg1 = sum(acc1s) / len(acc1s)
        logger.info(f"Average acc1s are {avg1}")
        acc5s.append(top5.item())
        logger.info(f"acc1s are {acc1s}")
        logger.info(f"acc5s are {acc5s}")
        if args.repeat:
            logger.info(f"Now, start to repeat!")
            repeat_model = deepcopy(adapt_net)
            repeat_model.eval()
            acc1s_repeat = []
            for repeat_c in common_corruptions:
                load_func = {10: load_cifar10c, 100: load_cifar100c}
                x_test, y_test = load_func[args.num_classes](10000, 5, "./data", False, [repeat_c])
                val_loader = {'x': x_test, 'y': y_test}
                top1, top5 = validate(val_loader, repeat_model, args, mode='eval', repeat=True)
                acc1s_repeat.append(top1.item())
                avg1 = sum(acc1s_repeat) / len(acc1s_repeat)
                if repeat_c == corrupt:
                    logger.info(f"Repeat Average are {avg1}")
                    del repeat_model
                    break
