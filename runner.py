from __future__ import division, print_function

import matplotlib
matplotlib.use('agg')

import os
import sys
import importlib

from dar_package.datasets import vaihingen, bing
from dar_package.models.drn_contours import DRNContours
from dar_package import pretrain, train_baseline, train_contours
from dar_package import eval_contours, eval_baseline
from dar_package.utils.train_utils import save_config, overwrite_config
import dar_package.config


def pretrain_vaihingen(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Pretrain vaihingen {}".format(split_num))
    checkpoint_path = pretrain.run(config['pretrain_vaihingen_{}'.format(split_num)],
                                   config,
                                   vaihingen.VaihingenDataset,
                                   DRNContours,
                                   split_num)
    overwrite_config('vaihingen_exp_{}'.format(split_num), 'restore', checkpoint_path)


def pretrain_bing(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Pretrain bing {}".format(split_num))
    checkpoint_path = pretrain.run(config['pretrain_bing_{}'.format(split_num)],
                                   config,
                                   bing.BingDataset,
                                   DRNContours,
                                   split_num)
    overwrite_config('bing_exp_{}'.format(split_num), 'restore', checkpoint_path)


def train_baseline_vaihingen(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Baseline vaihingen {}".format(split_num))
    checkpoint_path = train_baseline.run(config['baseline_vaihingen_{}'.format(split_num)],
                                         config,
                                         vaihingen.VaihingenDataset,
                                         DRNContours,
                                   split_num)
    overwrite_config('baseline_vaihingen_{}'.format(split_num), 'eval_model', checkpoint_path)


def train_baseline_bing(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Baseline bing {}".format(split_num))
    checkpoint_path = train_baseline.run(config['baseline_bing_{}'.format(split_num)],
                                         config,
                                         bing.BingDataset,
                                         DRNContours,
                                   split_num)
    overwrite_config('baseline_bing_{}'.format(split_num), 'eval_model', checkpoint_path)


def train_vaihingen(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Train vaihingen {}".format(split_num))
    checkpoint_path = train_contours.run(config['vaihingen_exp_{}'.format(split_num)],
                                         config,
                                         vaihingen.VaihingenDataset,
                                         DRNContours,
                                         split_num)
    overwrite_config('vaihingen_exp_{}'.format(split_num), 'eval_model', checkpoint_path)


def train_bing(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    print("Train bing {}".format(split_num))
    checkpoint_path = train_contours.run(config['bing_exp_{}'.format(split_num)],
                                         config,
                                         bing.BingDataset,
                                         DRNContours,
                                         split_num)
    overwrite_config('bing_exp_{}'.format(split_num), 'eval_model', checkpoint_path)


def eval_baseline_vaihingen(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    save_dir = eval_baseline.run(config['baseline_vaihingen_{}'.format(split_num)],
                                 vaihingen.VaihingenDataset,
                                 DRNContours)


def eval_vaihingen(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    save_dir = eval_contours.run(config['vaihingen_exp_{}'.format(split_num)],
                                 vaihingen.VaihingenDataset,
                                 DRNContours)


def eval_baseline_bing(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    save_dir = eval_baseline.run(config['baseline_bing_{}'.format(split_num)],
                                 bing.BingDataset,
                                 DRNContours)


def eval_bing(split_num=1):
    importlib.reload(dar_package.config)
    config = dar_package.config.config
    save_dir = eval_contours.run(config['bing_exp_{}'.format(split_num)],
                                 bing.BingDataset,
                                 DRNContours)


def run_vaihingen(split_num):
    train_baseline_vaihingen(split_num)
    pretrain_vaihingen(split_num)
    train_vaihingen(split_num)
    eval_baseline_vaihingen(split_num)
    eval_vaihingen(split_num)


def run_bing(split_num):
    train_baseline_bing(split_num)
    pretrain_bing(split_num)
    train_bing(split_num)
    eval_baseline_bing(split_num)
    eval_bing(split_num)


def main(args):
    experiments = {'vaihingen': run_vaihingen,
                   'bing': run_bing}
    config = dar_package.config.config

    for dataset, coordinator in experiments.items():
        for split_num in range(1, int(config[dataset]['num_folds']) + 1):
            coordinator(split_num)


if __name__ == "__main__":
    main(sys.argv[1:])
