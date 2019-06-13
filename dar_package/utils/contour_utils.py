from __future__ import division, print_function

import torch
import numpy as np
import time

from dar_package.utils.data_utils import batch_diag


def evolve_active_rays_fast(rho, beta, data, kappa, theta, delta_theta, rho_target, origin,
                            delta_t=1e-4, max_steps=128, rho_min=1.0, debug_hist=False):

    assert (rho.device == beta.device == data.device 
            == theta.device == delta_theta.device == rho_target.device
            == origin.device)
    batch_size, height, width = data.size()
    _, L = rho.size()
    rho_init = rho.clone()
    rho_min = torch.ones_like(rho_init) * rho_min

    cos_theta = torch.cos(theta)                    # (N, L)
    sin_theta = torch.sin(theta)                    # (N, L)
    cos_delta_theta = torch.cos(delta_theta)        # (N, L)
    cos_2_delta_theta = torch.cos(2 * delta_theta)  # (N, L)
    idx = np.arange(L)                              # (L, )
    diagonal_length = np.linalg.norm([height, width])
    
    roll_m1 = np.roll(idx, -1)
    roll_p1 = np.roll(idx, 1)

    if debug_hist:
        rho_hist = torch.zeros(max_steps + 1, batch_size, L, device=rho.device, dtype=rho.dtype)
        rho_hist[0] = rho_init

    # Have to evaluate derivatives of alpha, beta, and data terms
    # with respect to rho (radius in polar coordinates). To do this,
    # have to evaluate in terms of cartesian coordinates (i.e., d/dx and d/dy)
    # first, and perform change of variable.
    sobel_kernel_x = torch.tensor([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
        ], device=data.device, requires_grad=False)
    sobel_kernel_y = torch.t(sobel_kernel_x)
    sobel_kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(1)
    sobel_kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(1)
    derivative_padding = (1, 1, 1, 1)
    d_data_dx = torch.nn.functional.conv2d(
        torch.nn.functional.pad(data.unsqueeze(1), derivative_padding, mode='replicate'),
        sobel_kernel_x).squeeze(1)      # (N, H, W)
    d_data_dy = torch.nn.functional.conv2d(
        torch.nn.functional.pad(data.unsqueeze(1), derivative_padding, mode='replicate'),
        sobel_kernel_y).squeeze(1)      # (N, H, W)

    batch_select_idx = torch.arange(batch_size).unsqueeze(1).repeat(1, L).flatten()

    for i in range(max_steps):
        # Coordinates are given as x, y (where coordinate origin is top left of frame, y is pointing down), so
        # need to extract the correct (row, column) of the maps through bilinear interpolation
        # origin: (N, 2)
        # rho: (N, L)
        # cos_theta: (N, L)
        # contour_x: (N, L) tensor of x coordinates
        contour_x = origin[:, 0].unsqueeze(1) + rho * cos_theta
        contour_y = origin[:, 1].unsqueeze(1) + rho * sin_theta

        # From these positions, get (N, L) tensors of interpolation anchors
        # Cap these values to the allowable regions of the map
        exceeded_left = contour_x < 0
        exceeded_right = contour_x > (width - 1)
        exceeded_top = contour_y < 0
        exceeded_bottom = contour_y > (height - 1)
        x1 = contour_x.floor().float()
        y1 = contour_y.floor().float()
        x1 = torch.max(x1, torch.zeros_like(x1))
        y1 = torch.max(y1, torch.zeros_like(y1))
        x1 = torch.min(x1, torch.ones_like(x1) * (width - 2))
        y1 = torch.min(y1, torch.ones_like(y1) * (height - 2))
        x2 = (x1 + 1).float()
        y2 = (y1 + 1).float()

        # Extract values from the corresponding maps for bilinear interpolation
        # x1, y1, x2, y2: (N, L) tensors each
        x1_idx = x1.long().flatten()
        x2_idx = x2.long().flatten()
        y1_idx = y1.long().flatten()
        y2_idx = y2.long().flatten()

        # Resultant tensors are (N, L)
        beta_Q11 = beta[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        beta_Q12 = beta[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        beta_Q21 = beta[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        beta_Q22 = beta[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        beta_i = (beta_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  beta_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  beta_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  beta_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))

        d_data_dx_Q11 = d_data_dx[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        d_data_dx_Q12 = d_data_dx[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        d_data_dx_Q21 = d_data_dx[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        d_data_dx_Q22 = d_data_dx[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        d_data_dx_i = (d_data_dx_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  d_data_dx_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  d_data_dx_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  d_data_dx_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))

        d_data_dy_Q11 = d_data_dy[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        d_data_dy_Q12 = d_data_dy[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        d_data_dy_Q21 = d_data_dy[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        d_data_dy_Q22 = d_data_dy[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        d_data_dy_i = (d_data_dy_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  d_data_dy_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  d_data_dy_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  d_data_dy_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))

        kappa_Q11 = kappa[batch_select_idx, y1_idx, x1_idx].view(batch_size, -1)
        kappa_Q12 = kappa[batch_select_idx, y2_idx, x1_idx].view(batch_size, -1)
        kappa_Q21 = kappa[batch_select_idx, y1_idx, x2_idx].view(batch_size, -1)
        kappa_Q22 = kappa[batch_select_idx, y2_idx, x2_idx].view(batch_size, -1)
        kappa_i = (kappa_Q11 * (x2 - contour_x) * (y2 - contour_y) +
                  kappa_Q21 * (contour_x - x1) * (y2 - contour_y) +
                  kappa_Q12 * (x2 - contour_x) * (contour_y - y1) + 
                  kappa_Q22 * (contour_x - x1) * (contour_y - y1)) / ((x2 - x1) * (y2 - y1))

        # Change of variable. Result should still be (N, L)
        d_data_d_rho_i = -(d_data_dx_i * cos_theta + d_data_dy_i * sin_theta)

        # In places where contour has exceeded image bounds, want to kill the evolution
        # or else contour may keep evolving further and further outwards, which will cause exploding
        # gradients (e.g., the edge value of a map gets repeatedly multiplied, causing the
        # contour to move outwards, which will cause a huge erroneous update)
        beta_i = beta_i.clone()
        d_data_dx_i = d_data_dx_i.clone()
        d_data_dy_i = d_data_dy_i.clone()
        kappa_i = kappa_i.clone()
        d_data_d_rho_i = d_data_d_rho_i.clone()
        beta_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0
        d_data_dx_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0
        d_data_dy_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0
        kappa_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0
        d_data_d_rho_i[exceeded_left | exceeded_right | exceeded_top | exceeded_bottom] = 0

        # Construct cyclic pentadiagonal matrix
        # a = beta_i[1:L-1] * cos_2_delta_theta / delta_theta**4
        # b = (-2 * beta_i[1:L] - 2 * beta_i[:L-1]) * cos_delta_theta / delta_theta**4
        # c = (beta_i[np.roll(idx, -1)] - 4 * beta_i + beta_i[np.roll(idx, 1)]) / delta_theta**4
        # d = b
        # e = a
        # a_corner = beta_i[np.roll(idx, -1)[-2:]] * cos_2_delta_theta / delta_theta**4
        # b_corner = (-2 * beta_i[0] - 2 * beta_i[-1]) * cos_delta_theta / delta_theta**4
        # e_corner = a_corner
        # d_corner = b_corner   (N, 1)
        a = beta_i[:, 1:L-1] * cos_2_delta_theta
        b = (-2 * beta_i[:, 1:L] - 2 * beta_i[:, :L-1]) * cos_delta_theta
        c = (beta_i[:, roll_m1] + 4 * beta_i + beta_i[:, roll_p1])
        d = b
        e = a
        a_corner = beta_i[:, roll_m1[-2:]] * cos_2_delta_theta
        b_corner = (-2 * beta_i[:, 0].unsqueeze(1) - 2 * beta_i[:, -1].unsqueeze(1)) * cos_delta_theta
        e_corner = a_corner
        d_corner = b_corner

        # Above entries are all (N, x)
        # Need to build batched diagonal matrices
        # Result: A is (N, L, L)
        A = (
            batch_diag(c) + batch_diag(b, diagonal=1) + batch_diag(d, diagonal=-1) + 
            batch_diag(a, diagonal=2) + batch_diag(e, diagonal=-2) + 
            batch_diag(e_corner, diagonal=(L - 2)) + batch_diag(a_corner, diagonal=-(L - 2)) + 
            batch_diag(d_corner, diagonal=(L - 1)) + batch_diag(b_corner, diagonal=-(L - 1))
        )

        # External energy term
        f = d_data_d_rho_i

        # Balloon
        g = -kappa_i

        # Normalize rho by diagonal length so we remain consistent with our balloon
        # term equation (realistically everything else just gets scaled, but keeping
        # rho small should provide some numeric stability)
        rho = rho / diagonal_length

        # Calculate gradient descent step:
        # A (N, L, L) @ rho_i (N, L) + f (N, L)
        # Do batch matrix-matrix product and addition; requires matrices (3D)
        # (N, L, L) @ (N, L, 1) ==> (N, L, 1) + (N, L, 1) ==> (N, L, 1) ==> (N, L)
        assert (A.device == rho.device == f.device)
        d_E_d_rho = torch.baddbmm((f + g).unsqueeze(2), A, rho.unsqueeze(2)).squeeze(2)
        step = d_E_d_rho * delta_t
        
        # Take the step
        rho = rho - step

        # Unnormalize
        rho = rho * diagonal_length

        if debug_hist:
            rho_hist[i + 1] = rho

    if debug_hist:
        return rho, rho_hist

    return rho
