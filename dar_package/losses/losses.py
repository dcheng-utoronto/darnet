from __future__ import division, print_function

import torch
import torch.nn as nn

from dar_package.utils.contour_utils import evolve_active_rays_fast


class DistanceLossFast(nn.Module):
    def __init__(self, delta_t=2e-4, max_steps=200):
        super(DistanceLossFast, self).__init__()
        self.delta_t = delta_t
        self.max_steps = max_steps

    def forward(self, rho_init, rho_target, origin,
        beta, data, kappa, 
        theta, delta_theta, debug_hist=False):
        result = evolve_active_rays_fast(
            rho_init, beta, data, kappa, 
            theta, delta_theta, rho_target, origin,
            delta_t=self.delta_t, max_steps=self.max_steps, debug_hist=debug_hist)

        if debug_hist:
            rho, rho_hist = result
        else:
            rho = result
        
        rho_diff = torch.nn.functional.l1_loss(rho, rho_target)

        with torch.no_grad():
            _, height, width = data.size()

            # Join two tensors of (N, L) to (N, L, 2)
            # Then add the origin (N, 1, 2) to broadcast
            rho_cos_theta = rho * torch.cos(theta)
            rho_sin_theta = rho * torch.sin(theta)
            joined = torch.stack([rho_cos_theta, rho_sin_theta], dim=2)
            contour = origin.unsqueeze(1) + joined
            contour_x = contour[..., 0]
            contour_y = contour[..., 1]

            if debug_hist:
                batch_size, hist_len, contour_len = rho_hist.size()
                rho_cos_theta = rho_target * torch.cos(theta)
                rho_sin_theta = rho_target * torch.sin(theta)
                joined = torch.stack([rho_cos_theta, rho_sin_theta], dim=2)
                contour_target = origin.unsqueeze(1) + joined
                rho_target_x = contour_target[..., 0]
                rho_target_y = contour_target[..., 1]

                # rho_hist is (N, H, L)
                # rho_hist_cos_theta is (N, H, L)
                # joined is (N, H, L, 2)
                rho_hist_cos_theta = rho_hist * torch.cos(theta).unsqueeze(1)
                rho_hist_sin_theta = rho_hist * torch.sin(theta).unsqueeze(1)
                rho_hist_joined = torch.stack([rho_hist_cos_theta, rho_hist_sin_theta], dim=3)
                rho_hist_xy = rho_hist_joined + origin.unsqueeze(1).unsqueeze(2)
                rho_hist_x = rho_hist_xy[..., 0]
                rho_hist_y = rho_hist_xy[..., 1]

                return rho_diff, contour_x, contour_y, rho, rho_hist_x, rho_hist_y, rho_target_x, rho_target_y

        return rho_diff, contour_x, contour_y, rho
