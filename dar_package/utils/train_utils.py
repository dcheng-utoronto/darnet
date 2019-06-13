from __future__ import division, print_function

import torch
import numpy as np
from PIL import Image
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import importlib
import dar_package.config


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Copied from example on PyTorch github
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format
    Copied from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # put the figure pixmap into a np array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def fig2data ( fig ):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Copied from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # draw the renderer
    fig.canvas.draw ()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def unpack_sample(sample):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for key, value in sample.items():
        if type(value) == torch.Tensor:
            sample[key] = value.to(device)
    return sample


def save_config(config, save_folder, filename='config.ini'):
    with open(os.path.join(save_folder, filename), 'w') as configfile:
        config.write(configfile)


def overwrite_config(dict_name, key, value, filename='config.ini'):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    config[dict_name][key] = value
    save_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                               "config")
    save_config(config, save_folder, filename=filename)
    

def plot_sample_eval_contours(image, mask, beta, data, kappa, init_x, init_y, final_x, final_y,
                sequence_id, save_folder, caption=None):
    height = 6
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(height * 5, height))
    ax[0].imshow(image)

    ax[1].imshow(image)
    gt_mask = np.zeros_like(image)
    gt_mask[:, :, 1] = mask
    ax[1].imshow(gt_mask, alpha=0.5)
    ax[1].plot(final_x, final_y, '--', lw=3, color=[1, 1, 0])
    ax[1].plot(init_x, init_y, '-.', lw=3, color=[0, 0, 1])
    if caption is not None:
        ax[1].set_title(caption)

    im = ax[2].imshow(data)
    ax[2].plot(final_x, final_y, '--', lw=3, color=[1, 1, 0])
    ax[2].plot(init_x, init_y, '-.', lw=3, color=[0, 0, 1])
    f.colorbar(im, ax=ax[2])
    ax[2].set_title("Data")

    im = ax[3].imshow(beta)
    ax[3].plot(final_x, final_y, '--', lw=3, color=[1, 1, 0])
    ax[3].plot(init_x, init_y, '-.', lw=3, color=[0, 0, 1])
    f.colorbar(im, ax=ax[3])
    ax[3].set_title("Beta")

    im = ax[4].imshow(kappa)
    ax[4].plot(final_x, final_y, '--', lw=3, color=[1, 1, 0])
    ax[4].plot(init_x, init_y, '-.', lw=3, color=[0, 0, 1])
    f.colorbar(im, ax=ax[4])
    ax[4].set_title("Kappa")

    plot_im = fig2img(f)
    plot_im.save(os.path.join(save_folder, "{}.png".format(sequence_id)))
    plt.close('all')


def vis_output(contour_x, contour_y, *maps, unnormalize_function=None):
    num_maps = len(maps)
    num_cols = 3
    num_rows = num_maps // num_cols
    if num_maps % num_cols > 0:
        num_rows += 1

    contour_x = contour_x[0].detach().squeeze().cpu().numpy()
    contour_y = contour_y[0].detach().squeeze().cpu().numpy()

    fig, _ = plt.subplots(num_rows, num_cols)
    for i, ax in enumerate(fig.axes):
        if i == len(maps):
            break
        map_to_plot = maps[i][0].detach().squeeze().cpu()
        if map_to_plot.dim() == 3 and unnormalize_function is not None:
            map_to_plot = unnormalize_function(map_to_plot)
        map_to_plot = map_to_plot.numpy()            

        if map_to_plot.ndim == 3:
            map_to_plot = map_to_plot.transpose(1, 2, 0)
        im = ax.imshow(map_to_plot)
        ax.scatter(contour_x, contour_y, s=2, c='red')
        fig.colorbar(im, ax=ax)

    return fig


def vis_maps(*maps, unnormalize_function=None):
    num_maps = len(maps)
    num_cols = 3
    num_rows = num_maps // num_cols
    if num_maps % num_cols > 0:
        num_rows += 1

    fig, axs = plt.subplots(num_rows, num_cols)
    for i, ax in enumerate(fig.axes):
        if i == len(maps):
            break
        map_to_plot = maps[i][0].detach().squeeze().cpu()
        if map_to_plot.dim() == 3 and unnormalize_function is not None:
            map_to_plot = unnormalize_function(map_to_plot)
        map_to_plot = map_to_plot.numpy()            

        if map_to_plot.ndim == 3:
            map_to_plot = map_to_plot.transpose(1, 2, 0)
        ax.imshow(map_to_plot)

    return fig
