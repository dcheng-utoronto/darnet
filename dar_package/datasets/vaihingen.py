from __future__ import print_function, division

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image

from dar_package.config import config
from dar_package.datasets.base_dataset import BaseDataset


class VaihingenDataset(BaseDataset):
    MEAN = np.array([0.47341759, 0.28791303, 0.2850705])
    STD = np.array([0.22645572, 0.15276193, 0.140702])
    ORIG_SIZE = 512
    ORIG_DIST_PER_PX = 0.09

    def __init__(self, split, final_size=256, discretize_points=60,
        init_contour_radius=20, normalize=True):
        self.cfg = config['vaihingen']
        super(VaihingenDataset, self).__init__(
            split, self.cfg['image_extension'], init_contour_radius,
            final_size=final_size, discretize_points=discretize_points,
            normalize=normalize)

        self.color_jitter = transforms.ColorJitter(brightness=0.6, contrast=0.5, 
            saturation=0.4, hue=0.025)
        self.normalize = transforms.Normalize(self.MEAN, self.STD)
        self.unnormalize = transforms.Normalize(-self.MEAN / self.STD, 1 / self.STD)

        image_paths = self.cfg[split]
        with open(image_paths, 'r') as f:
            for line in f:
                self.image_paths.append(line.strip())

        gt_csv = self.cfg['gt_polygon_file']
        df = pd.read_csv(gt_csv, header=None)
        for path in self.image_paths:
            original_size = Image.open(path).size
            assert original_size[0] == original_size[1], "Only square images supported"

            sequence_id = int(self.image_path_to_sequence_id(path))

            # First entry of row denotes number of vertices
            # Polygon is given with last vertex repeated
            num_vertices = df.iloc[sequence_id - 1][0]

            # Polygon is given as a series of {(r1, c1), (r2, c2), ...}
            # Read into array with columns of r, c
            # Note that these are not (x, y) coordinates, but (row, col), so transform them
            # by swapping columns
            polygon = np.array(df.iloc[sequence_id - 1][1:num_vertices * 2 + 1].tolist()).reshape(num_vertices, 2)
            polygon[:, [1, 0]] = polygon[:, [0, 1]]
            
            # Adjust coordinates to target output size
            polygon = polygon * self.final_size / original_size
            self.gt_polygons.append(polygon)

    def get_image(self, idx):
        return self.get_image_base(idx, 2)

    def get_masks(self, idx):
        image_ending = os.path.basename(self.image_paths[idx])
        mask_ending_all = image_ending.replace("building_", "all_buildings_mask_")
        mask_ending_one = image_ending.replace("building_", "building_mask_")
        mask_path_all = self.image_paths[idx].replace(image_ending, mask_ending_all)
        mask_path_one = self.image_paths[idx].replace(image_ending, mask_ending_one)
        image = Image.open(mask_path_all)
        mask_all = transforms.functional.resize(image, self.final_size)
        image = Image.open(mask_path_one)
        mask_one = transforms.functional.resize(image, self.final_size)
        return mask_all, mask_one

    def __getitem__(self, idx):
        image = self.get_image(idx)
        polygon = self.gt_polygons[idx].copy()
        mask_all, mask_one = self.get_masks(idx)
        sequence_id = int(self.image_path_to_sequence_id(self.image_paths[idx]))

        # Data augmentation
        if 'train' in self.split:
            mask_collection = [mask_all, mask_one]
            image, mask_collection, polygon = self.random_flips(image, mask_collection, polygon)
            image, mask_collection, polygon = self.random_scale(image, mask_collection, polygon)

            # Randomly rotate image, with resolution of 0.01 radians
            rotation_angle = np.round(np.random.uniform(0, 2 * np.pi), decimals=2)
            image, mask_collection, polygon = self.rotate_image_masks_poly(
                image, mask_collection, polygon, rotation_angle)           
                
            mask_all, mask_one = mask_collection
            image = self.color_jitter(image)

        semantic_mask = self.get_semantic_mask(mask_all)

        # Compute distance transforms
        distance_transform_mask = mask_all
        distance_transform = self.get_distance_transform(distance_transform_mask)
        distance_transform_inside = self.get_distance_transform(distance_transform_mask, side='inside')
        distance_transform_outside = self.get_distance_transform(distance_transform_mask, side='outside')

        # Get the initial contour origin
        if 'train' in self.split:
            init_contour_origin = self.initialize_contour_origin(mask_one)
        else:
            init_contour_origin = self.midpoint

        # Get polygons for ground truth and initialization
        interp_coords, interp_radii, interp_angles, delta_angles = self.interpolate_ground_truth_polygon(
            polygon, init_contour_origin)
        init_contour_radii = self.initialize_contour_radii(interp_angles)

        # Get the initialized contour distance transform
        init_contour_dt = self.generate_init_distance_transform(
            init_contour_origin, init_contour_radii, interp_angles)

        # Get interpolated ground truth polygon
        gt_snake = self.interpolate_gt_snake(polygon)
        gt_snake_x = gt_snake[:, 0]
        gt_snake_y = gt_snake[:, 1]

        # Transform to torch tensors
        image = transforms.functional.to_tensor(image)
        mask_all = transforms.functional.to_tensor(mask_all)
        mask_one = transforms.functional.to_tensor(mask_one)
        semantic_mask = torch.from_numpy(semantic_mask).long()
        distance_transform = torch.from_numpy(distance_transform).float()
        distance_transform_inside = torch.from_numpy(distance_transform_inside).float()
        distance_transform_outside = torch.from_numpy(distance_transform_outside).float()
        interp_coords = torch.from_numpy(interp_coords).float()
        interp_radii = torch.from_numpy(interp_radii).float()
        interp_angles = torch.from_numpy(interp_angles).float()
        delta_angles = torch.Tensor([delta_angles]).float()
        init_contour_radii = torch.from_numpy(init_contour_radii).float()
        init_contour_origin = torch.from_numpy(init_contour_origin).float()
        gt_snake_x = torch.from_numpy(gt_snake_x).float()
        gt_snake_y = torch.from_numpy(gt_snake_y).float()

        # Normalize image
        if self.normalize_flag:
            image = self.normalize(image)

        # NOTE: There is no nice way to include the ground truth polygon because 
        # they have variable length, so they cannot be placed into batches; 
        # include sequence id instead to have back reference
        sample = {
            'image': image,
            'mask': mask_all,
            'mask_one': mask_one,
            'semantic_mask': semantic_mask,
            'distance_transform': distance_transform,
            'distance_transform_inside': distance_transform_inside,
            'distance_transform_outside': distance_transform_outside,
            'interp_poly': interp_coords,
            'interp_radii': interp_radii,
            'interp_angles': interp_angles,
            'init_contour': init_contour_radii,
            'init_contour_origin': init_contour_origin,
            'delta_angles': delta_angles,
            'sequence_id': sequence_id,
            'gt_snake_x': gt_snake_x,
            'gt_snake_y': gt_snake_y
        }
        return sample