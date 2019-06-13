from __future__ import print_function, division

import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import skimage.transform
import warnings
import copy

from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import binary_erosion, binary_dilation
from scipy import interpolate

from dar_package.utils.data_utils import draw_circle, draw_poly_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, split, image_ext, init_contour_radius,
        final_size=256, discretize_points=60, normalize=True):
        self.split = split
        self.final_size = final_size
        self.discretize_points = discretize_points
        self.midpoint = np.array([final_size // 2, final_size // 2])
        self.image_paths = []
        self.gt_polygons = []
        self.image_ext = image_ext
        self.init_contour_radius = init_contour_radius
        self.normalize_flag = normalize
        self.crop = transforms.CenterCrop(self.final_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_masks(self, idx):
        raise NotImplementedError

    def image_path_to_sequence_id(self, path):
        sequence_id = path.split('/')[-1].strip().replace(self.image_ext, '').split('_')[-1]
        return sequence_id

    def get_image_base(self, idx, interpolation):
        image = Image.open(self.image_paths[idx])
        image = transforms.functional.resize(image, self.final_size, interpolation)
        return image

    def get_distance_transform(self, mask, side='both'):
        """ Distance transform to the ground truth mask, both inside and outside the building.
        """
        mask = np.asarray(mask)
        dist_trans_inside = distance_transform_edt(mask).astype(float)
        if side == 'inside':
            return dist_trans_inside
        dist_trans_outside = distance_transform_edt(np.logical_not(mask)).astype(float)
        if side == 'outside':
            return dist_trans_outside
        dist_trans = dist_trans_inside + dist_trans_outside
        return dist_trans

    def get_edge_mask(self, mask, edge_width=5):
        dilated = np.asarray(mask)

        for _ in range(edge_width):
            dilated = binary_dilation(dilated)

        edges = dilated ^ mask
        edges = edges.astype(np.uint8) * 255
        edges = Image.fromarray(edges).convert('1')
        return edges

    def get_semantic_mask(self, mask, edge_width=5):
        """ Mask for semantic segmentation.

            Key:
            0: Background
            1: Building
            2: Boundary
        """
        edge_mask = np.asarray(self.get_edge_mask(mask, edge_width=edge_width))
        mask = np.asarray(mask)
        semantic_mask = np.asarray(mask) * 0
        semantic_mask = semantic_mask.astype(np.uint8)
        semantic_mask[mask == 0] = 0
        semantic_mask[mask == 1] = 1
        semantic_mask[edge_mask == 1] = 2
        return semantic_mask

    def rotate_image_masks_poly(self, image, masks_to_rotate, poly, angle):
        """ Rotate the image, mask, and initial polygon around the midpoint
        """
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Translate polygon around its midpoint to rotate the coordinates
        poly = poly - self.midpoint
        poly = poly.dot(R)
        poly = poly + self.midpoint

        image = np.asarray(image)
        image = skimage.transform.rotate(image, angle * 180 / np.pi, order=3, mode='reflect')
        image = Image.fromarray((image * 255).astype(np.uint8))

        rotated_masks = []
        for mask in masks_to_rotate:
            mask_np = np.uint8(np.asarray(mask) * 255)
            mask_np = skimage.transform.rotate(mask_np, angle * 180 / np.pi, order=3, mode='reflect')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask_np = skimage.img_as_ubyte(mask_np)
            mask_pil = Image.fromarray(mask_np)
            rotated_masks.append(mask_pil)
        rotated_masks = [mask.convert('1') for mask in rotated_masks]

        return image, rotated_masks, poly

    def poly_cart_to_polar(self, poly, origin=None):
        if origin is None:
            origin = self.midpoint

        # Center polygon on new origin
        poly_cart = poly.copy()
        poly_cart -= origin

        # Get radii and angles
        # Remember arctan2 takes y,x coordinates, and can output negative angles
        poly_r = np.linalg.norm(poly_cart, axis=1)
        poly_theta = np.arctan2(poly_cart[:, 1], poly_cart[:, 0])
        poly_theta = (poly_theta + 2 * np.pi) % (2 * np.pi)
        return poly_r, poly_theta, origin

    def poly_polar_to_cart(self, poly_r, poly_theta, origin):
        poly_x = poly_r * np.cos(poly_theta)
        poly_y = poly_r * np.sin(poly_theta)
        poly_cart = np.hstack((poly_x[:, np.newaxis], poly_y[:, np.newaxis]))
        poly_cart += origin
        return poly_cart

    def random_flips(self, image, mask_collection, polygon):
        flipped_masks = copy.deepcopy(mask_collection)

        # Random flip image up/down
        if np.random.binomial(1, 0.5) == 1:
            image = transforms.functional.vflip(image)
            for i in range(len(flipped_masks)):
                flipped_masks[i] = transforms.functional.vflip(flipped_masks[i])
            polygon[:, 1] -= self.midpoint[1]
            polygon[:, 1] *= -1
            polygon[:, 1] += self.midpoint[1]
        
        # Random flip image left/right
        if np.random.binomial(1, 0.5) == 1:
            image = transforms.functional.hflip(image)
            for i in range(len(flipped_masks)):
                flipped_masks[i] = transforms.functional.hflip(flipped_masks[i])
            polygon[:, 0] -= self.midpoint[0]
            polygon[:, 0] *= -1
            polygon[:, 0] += self.midpoint[0]

        return image, flipped_masks, polygon

    def dataset_mean_std(self):
        rgb_channels = []
        for path in self.image_paths:
            image = np.asarray(Image.open(path))
            rgb_channels.append(image.reshape(-1, 3))
        rgb_channels = np.vstack(rgb_channels)
        mean = np.mean(rgb_channels, axis=0) / 255
        std = np.std(rgb_channels, axis=0) / 255
        return mean, std

    def interpolate_gt_snake(self, poly):
        interp_poly = np.zeros((self.discretize_points, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tck, u = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
            interp_poly[:, 0], interp_poly[:, 1] = interpolate.splev(np.linspace(0, 1, self.discretize_points), tck)
        return interp_poly

    def random_scale(self, image, masks_to_scale, polygon, padding_mode=['reflect','constant'],
                     upper_bound=0.75, lower_bound=0.3):
        """ Randomly scale the image so that the polygon stays within image bounds
        """
        poly_r, poly_theta, origin = self.poly_cart_to_polar(polygon)
        max_r = poly_r.max()
        min_r = poly_r.min()

        # Set absolute bounds (in pixels) of how much we can scale up/down the image
        # around the center. If the existing polygon already exceeds these bounds, then
        # use these as the bounds
        half_distance = self.final_size // 2
        upper_bound_r = upper_bound * half_distance
        lower_bound_r = lower_bound * half_distance
        upper_bound_r = max(upper_bound_r, max_r)
        lower_bound_r = min(lower_bound_r, min_r)

        # Select a scale according to these bounds
        # scale_up: At least 1
        # scale_down: At most 1
        scale_up = upper_bound_r / max_r
        scale_down = lower_bound_r / min_r
        scale_factor = np.random.uniform(scale_down, scale_up)
        scaled_size = int(np.round(half_distance * scale_factor * 2, decimals=0))
        scaled_size = scaled_size // 2 * 2
        scale_factor = scaled_size / self.final_size
        scaled_masks = []
        image = transforms.functional.resize(image, scaled_size)

        # Have to do this strange thing with PIL so it will downside binary images correctly
        for mask in masks_to_scale:
            mask_pil = Image.fromarray(np.uint8(np.asarray(mask) * 255))
            downsized = transforms.functional.resize(mask_pil, scaled_size)
            scaled_masks.append(downsized)

        if scaled_size > self.final_size:
            image = self.crop(image)
            scaled_masks = [self.crop(mask) for mask in scaled_masks]
        elif scaled_size < self.final_size:
            padding = (self.final_size - scaled_size) // 2
            image = transforms.functional.pad(image, padding, padding_mode='reflect')
            padded_masks = []
            for idx, mask in enumerate(scaled_masks):
                padded_masks.append(transforms.functional.pad(mask, padding, padding_mode=padding_mode[idx]))
            scaled_masks = padded_masks

        # More wonky stuff with PIL
        scaled_masks = [mask.convert('1') for mask in scaled_masks]

        # Finally, scale the polygon coordinates themselves
        poly_r *= scale_factor
        poly = self.poly_polar_to_cart(poly_r, poly_theta, origin)
        return image, scaled_masks, poly

    def interpolate_ground_truth_polygon(self, poly, origin=None):
        """ Given ground truth polygon, interpolate points for ground truth active rays.
        """
        angles, delta_angles = np.linspace(0, 2 * np.pi, self.discretize_points, endpoint=False, retstep=True)
        interp_coords = []
        interp_radii = []
        if origin is None:
            origin = self.midpoint

        for theta in angles:
            dy = np.sin(theta)
            dx = np.cos(theta)
            d = np.array([dx, dy])
            v3 = np.array([-dy, dx])

            a = poly[:-1, :]
            b = poly[1:, :]
            v1 = a - origin
            v2 = a - b
            
            t2 = np.inner(v1, v3) / np.inner(v2, v3)
            t1 = np.cross(v1, -v2) / np.inner(-v2, v3)

            intersections = np.logical_and(t1 > 0, np.logical_and(0 <= t2, t2 <= 1))
            radii = t1 * intersections
            radii[np.isnan(radii)] = 0
            
            if np.count_nonzero(radii) > 0:
                radius = np.min(radii[radii.nonzero()])
                interp_radii.append(radius)
                interp_coords.append(origin + radius * d)

        assert len(interp_radii) == self.discretize_points, "Radii is {}".format(interp_radii)
                
        interp_coords = np.array(interp_coords)
        interp_radii = np.array(interp_radii)
        return interp_coords, interp_radii, angles, delta_angles

    def initialize_contour_radii(self, interp_angles):
        contour_radii = np.ones_like(interp_angles) * self.init_contour_radius
        return contour_radii

    def initialize_contour_origin(self, mask_one):
        # Erode the binary mask a number of times so we avoid initializing near boundaries
        positions = np.asarray(mask_one)
        for _ in range(self.init_contour_radius + 5):
            eroded = binary_erosion(positions)
            if np.count_nonzero(eroded) == 0:
                break
            positions = eroded

        nonzero_indices = np.argwhere(positions)
        pos_row, pos_col = nonzero_indices[np.random.choice(nonzero_indices.shape[0])]
        contour_origin = np.array([pos_col, pos_row])
        return contour_origin

    def generate_init_distance_transform(self, init_contour_origin, init_contour_radii, interp_angles):
        # Reconstruct the initial contour
        init_cart = self.poly_polar_to_cart(init_contour_radii, interp_angles, init_contour_origin)
        init_mask = draw_poly_mask(init_cart[:, 0], init_cart[:, 1], (self.final_size, self.final_size))
        distance_transform = self.get_distance_transform(init_mask, side='inside')
        return distance_transform