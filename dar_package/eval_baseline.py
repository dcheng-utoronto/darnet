from __future__ import division, print_function

import torch
import os
import datetime
import shutil
import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import skimage

from dar_package.config import config
from dar_package.utils.train_utils import unpack_sample, fig2img
from dar_package.utils.data_utils import compute_iou
from dar_package.train_baseline import ModelAndLoss
from dar_package.utils.eval_utils import db_eval_boundary


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
                _, output = model_and_loss(sample)
            _, _, _, combined = output
            
            for j in range(combined.size()[0]):
                predict_mask = (combined[j] == 1)
                predict_mask = predict_mask.squeeze().cpu().numpy()
                predict_mask = flood_prediction(predict_mask)
                gt_mask = sample['mask_one'][j].detach().squeeze().cpu().numpy()
                intersection, union, iou = compute_iou(predict_mask, gt_mask)
                running_intersection += intersection
                running_union += union
                example_iou += iou

                sequence_id = sample['sequence_id'][j].cpu().item()
                text = "Example {}: {}".format(sequence_id, iou)
                print(text)
                f.write(text + "\n")

                plot_sample(
                    dataset.unnormalize(sample['image'][j].squeeze()).detach().cpu().numpy().transpose(1, 2, 0),
                    sample['mask_one'][j].detach().squeeze().cpu().numpy(),
                    predict_mask,
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


def flood_prediction(predict_mask):
    # Copying the same evaluation method as DSAC
    predict_mask = predict_mask.astype(np.int32)
    size, _ = predict_mask.shape
    g = np.abs(np.linspace(-1, 1, size))
    G0, G1 = np.meshgrid(g, g)
    d = (1 - np.sqrt(G0 * G0 + G1 * G1))
    val = np.max(d * predict_mask)
    seed_im = np.int32(d * predict_mask == val)
    if val > 0:
        predict_mask = skimage.morphology.reconstruction(seed_im, predict_mask)
    return predict_mask


def plot_sample(image, mask, predict_mask,
                sequence_id, save_folder, caption=None):
    height = 6
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(height * 3, height))
    ax[0].imshow(image)

    ax[1].imshow(image)
    gt_mask = np.zeros_like(image)
    gt_mask[:, :, 1] = mask
    ax[1].imshow(gt_mask, alpha=0.5)

    ax[2].imshow(image)
    pr_mask = np.zeros_like(image)
    pr_mask[:, :, 0] = predict_mask
    pr_mask[:, :, 1] = predict_mask
    ax[2].imshow(pr_mask, alpha=0.5)
    if caption is not None:
        ax[2].set_title(caption)

    plot_im = fig2img(f)
    plot_im.save(os.path.join(save_folder, "{}.png".format(sequence_id)))
    plt.close('all')
