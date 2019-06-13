from __future__ import division, print_function

import torch
import os
import datetime
import shutil
import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from dar_package.config import config
from dar_package.utils.train_utils import unpack_sample, plot_sample_eval_contours
from dar_package.utils.data_utils import draw_poly_mask, compute_iou
from dar_package.utils.eval_utils import db_eval_boundary
from dar_package.train_contours import ModelAndLoss


def run(cfg, Dataset, Network):
    restore = cfg['eval_model']
    time = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    exp_id = "eval_{}_{}".format(cfg['name'], time)
    save_folder = os.path.join(os.path.dirname(restore), exp_id)

    os.mkdir(save_folder)
    results_text = os.path.join(save_folder, "results.txt")
    print("Creating {}".format(save_folder))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_and_loss = ModelAndLoss(Network, restore).to(device)
    model_and_loss.eval()

    dataset = Dataset(split='test')
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=int(cfg['batch_size']), 
        num_workers=int(cfg['num_workers']),
        shuffle=False)

    running_intersection = 0
    running_union = 0
    example_iou = 0

    f_bound_n_fg = [0] * 5
    f_bound_fg_match = [0] * 5
    f_bound_gt_match= [0] * 5
    f_bound_n_gt = [0] * 5

    with open(results_text, 'w') as f:
        for i, sample in enumerate(dataloader):
            with torch.no_grad():
                unpack_sample(sample)
                rho_diff, _, new_rho_x, new_rho_y, init_x, init_y, output = model_and_loss(sample)
                beta2, data2, kappa2 = output

            for j in range(new_rho_x.shape[0]):
                predict_mask = draw_poly_mask(
                    new_rho_x[j].detach().squeeze().cpu().numpy(),
                    new_rho_y[j].detach().squeeze().cpu().numpy(),
                    (dataset.final_size, dataset.final_size),
                    outline=1
                )
                gt_mask = sample['mask_one'][j].detach().squeeze().cpu().numpy()
                intersection, union, iou = compute_iou(predict_mask, gt_mask)
                running_intersection += intersection
                running_union += union
                example_iou += iou

                sequence_id = sample['sequence_id'][j].cpu().item()
                text = "Example {}: {}".format(sequence_id, iou)
                print(text)
                f.write(text + "\n")

                plot_sample_eval_contours(
                    dataset.unnormalize(sample['image'][j].squeeze()).detach().cpu().numpy().transpose(1, 2, 0),
                    sample['mask_one'][j].detach().squeeze().cpu().numpy(),
                    beta2[j].detach().squeeze().cpu().numpy(),
                    data2[j].detach().squeeze().cpu().numpy(),
                    kappa2[j].detach().squeeze().cpu().numpy(),
                    init_x[j].detach().cpu().numpy(),
                    init_y[j].detach().cpu().numpy(),
                    new_rho_x[j].detach().squeeze().cpu().numpy(),
                    new_rho_y[j].detach().squeeze().cpu().numpy(),
                    sequence_id,
                    save_folder,
                    caption="IOU = {}".format(iou)
                )

                for bounds in range(5):
                    _, _, _, fg_match, n_fg, gt_match, n_gt = db_eval_boundary(predict_mask, gt_mask, bound_th=bounds + 1)
                    f_bound_fg_match[bounds] += fg_match
                    f_bound_n_fg[bounds] += n_fg
                    f_bound_gt_match[bounds] += gt_match
                    f_bound_n_gt[bounds] += n_gt

        example_iou /= len(dataset)
        text = "mIOU: {}".format(example_iou)
        print(text)
        f.write(text + "\n")

        f_bound = [None] * 5
        for bounds in range(5):
            precision = f_bound_fg_match[bounds] / f_bound_n_fg[bounds]
            recall = f_bound_gt_match[bounds] / f_bound_n_gt[bounds]
            f_bound[bounds] = 2 * precision * recall / (precision + recall)

        text = ""
        for bounds in range(5):
            text += "F({})={},".format(bounds + 1, f_bound[bounds])
        text += "F(avg) = {}\n".format(sum(f_bound) / 5)
        f.write(text)

    return save_folder
