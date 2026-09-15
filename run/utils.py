"""Model construction, logging and instance visualization."""
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils


def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """Build the SAM2 mask path used by CA-SAM2 and StainPMS."""
    if net != "sam2":
        raise ValueError(f"Unsupported network: {net}")

    from sam2_train.build_sam import build_sam2

    net = build_sam2(args.sam_config, args.sam_ckpt, device=gpu_device)

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now().astimezone()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # Visualization output directory
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random()*255 for i in range(3)]
    return r, g, b


def get_inst_image(pred_masks): # 输入1，3，256，256
    pred_labeled = pred_masks.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
    pred_colored = np.zeros((pred_masks.shape[2], pred_masks.shape[3], 3))
    pred_labeled_cnum = pred_labeled.max() + 1
    for k in range(1, pred_labeled_cnum):
        pred_colored[pred_labeled == k, :] = np.array(get_random_color())
    return torch.tensor(pred_colored).unsqueeze(0).permute(0, 3, 1, 2)


def vis_inst_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    b, c, h, w = pred_masks.size()
    row_num = min(b, 4)

    imgs = torchvision.transforms.Resize((h, w))(imgs)
    if imgs.size(1) == 1:
        imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(imgs.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(imgs.device)

    imgs = imgs * std + mean

    pred_masks = get_inst_image(pred_masks).to(imgs.device)
    gt_masks = get_inst_image(gt_masks).to(imgs.device)
    pred_masks = pred_masks / 255.0
    gt_masks = gt_masks / 255.0

    # 每个 batch 取前 row_num 个样本
    imgs_vis = imgs[:row_num]
    pred_vis = pred_masks[:row_num]
    gt_vis = gt_masks[:row_num]

    # 纵向堆叠：img, pred, gt  -> 横向一行展示
    compose = torch.cat((imgs_vis, pred_vis, gt_vis), 0)  # total = 3 * row_num
    vutils.save_image(compose, fp=save_path, nrow=3 * row_num, padding=10)

    return
