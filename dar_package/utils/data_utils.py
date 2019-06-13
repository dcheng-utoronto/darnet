from PIL import Image, ImageDraw
import numpy as np
import torch


def draw_poly_mask(poly_x, poly_y, im_shape, outline=0):
    image = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    poly_coords = np.empty(poly_x.size + poly_y.size, dtype=poly_x.dtype)
    poly_coords[0::2] = poly_x
    poly_coords[1::2] = poly_y
    d.polygon(poly_coords.tolist(), fill=1, outline=outline)
    return np.array(image)


def draw_circle(image, center, radius, **kwargs):
    center_x, center_y = center
    draw = ImageDraw.Draw(image)
    draw.ellipse((center_x - radius, center_y - radius,
                  center_x + radius, center_y + radius),
                  **kwargs)


def batch_diag(bvec, diagonal=0):
    D = bvec.size(-1) + abs(diagonal)
    shape = (bvec.shape[0], D, D)
    device, dtype = bvec.device, bvec.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1, offset=diagonal)
    result.view(-1)[indices] = bvec
    return result


def compute_iou(predict_mask, gt_mask):
    intersection = np.count_nonzero(
        np.logical_and(predict_mask, gt_mask)
    )
    union = np.count_nonzero(
        np.logical_or(predict_mask, gt_mask)
    )
    return intersection, union, intersection / union