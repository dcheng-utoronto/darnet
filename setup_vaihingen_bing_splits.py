from __future__ import print_function, division

import os
import glob
import random
import argparse

from dar_package.config import config

random.seed(1234)


def enumerate_train_test(filenames, num_train):
    all_train_examples = []
    all_test_examples = []

    # DSAC does not shuffle filenames, just cut them off at the
    # right numbers
    for i, name in enumerate(filenames):
        if i < num_train:
            all_train_examples.append(name)
        else:
            all_test_examples.append(name)

    return all_train_examples, all_test_examples


def generate_split_files(all_train, all_test, num_val, num_folds, cfg):
    # Test split is set
    with open(cfg['test'], 'w') as test:
        for name in all_test:
            test.write("{}\n".format(name))

    # For train split, shuffle them
    random.shuffle(all_train)

    for i in range(num_folds):
        val_filenames = all_train[
            -(i + 1) * num_val : -i * num_val if i > 0 else None
        ]
        train_filenames = (
            all_train[:-(i + 1) * num_val] + all_train[-i * num_val:]
            if i > 0 else all_train[:-(i + 1) * num_val]
        )
        assert not set(train_filenames) & set(val_filenames)

        train_filenames.sort()
        val_filenames.sort()

        with open(cfg['train_{}'.format(i + 1)], 'w') as train:
            for name in train_filenames:
                train.write("{}\n".format(name))

        with open(cfg['val_{}'.format(i + 1)], 'w') as val:
            for name in val_filenames:
                val.write("{}\n".format(name))


def main(args):
    cfg = config[args.dataset_name[0]]
    num_train = int(cfg['num_train'])
    num_val_from_train = int(cfg['num_val_from_train'])
    num_folds = int(cfg['num_folds'])
    assert num_train // num_val_from_train == num_folds

    # Keep the original train/test division as in DSAC
    all_train, all_test = enumerate_train_test(
        sorted(glob.glob(cfg['image_glob'])), num_train)

    # Need to subdivide the train data further into splits
    generate_split_files(all_train, all_test, num_val_from_train, num_folds, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate splits')
    parser.add_argument('dataset_name',
                        nargs=1,
                        choices=['vaihingen', 'bing'],
                        help='vaihingen or bing')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
