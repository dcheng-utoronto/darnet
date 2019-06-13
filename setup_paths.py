import configparser
import os
import glob


def set_configs():
    dir_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    dir_config_file = './setup.ini'
    dir_config.read(dir_config_file)
    dir_sect = 'important_paths'

    package_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    package_config_file = './dar_package/config/config.ini'
    package_config.read(package_config_file)

    for section in package_config.sections():
        if section == 'vaihingen':
            package_config.set(section, 'data_path',
                dir_config.get(dir_sect, 'vaihingen_dataset_dir'))
            for i in range(1, int(package_config.get(section, 'num_folds')) + 1):
                package_config.set(section, 'train_{}'.format(i),
                    '${{data_path}}/train_split_{}.txt'.format(i))
                package_config.set(section, 'val_{}'.format(i),
                    '${{data_path}}/val_split_{}.txt'.format(i))
        elif section == 'bing':
            package_config.set(section, 'data_path',
                dir_config.get(dir_sect, 'bing_dataset_dir'))
            for i in range(1, int(package_config.get(section, 'num_folds')) + 1):
                package_config.set(section, 'train_{}'.format(i),
                    '${{data_path}}/train_split_{}.txt'.format(i))
                package_config.set(section, 'val_{}'.format(i),
                    '${{data_path}}/val_split_{}.txt'.format(i))
        else:
            package_config.set(section, 'save_path',
                dir_config.get(dir_sect, 'result_save_dir'))
            if package_config.has_option(section, 'eval_model'):
                package_config.set(section, 'eval_model', '/path/to/trained/model')
            if package_config.has_option(section, 'restore'):
                package_config.set(section, 'restore', '/path/to/pretrained/model')

    with open(package_config_file, 'w') as f:
        package_config.write(f)


def unify_bing_naming_scheme():
    """Pad bing images with 3-wide zeros"""
    package_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    package_config_file = './dar_package/config/config.ini'
    package_config.read(package_config_file)
    bing_dir = package_config.get('bing', 'data_path')
    image_extension = package_config.get('bing', 'image_extension')
    
    all_image_paths = glob.glob(os.path.join(bing_dir, '*{}'.format(image_extension)))

    for path in all_image_paths:
        orig_base = os.path.basename(path)
        orig_base = orig_base.replace(image_extension, '').split('_')
        sequence_id = int(orig_base[-1])
        new_base = orig_base[:-1] + ['{:03}'.format(sequence_id)]
        new_base = '_'.join(new_base) + image_extension
        new_path = os.path.join(os.path.dirname(path), new_base)
        os.rename(path, new_path)


if __name__ == '__main__':
    set_configs()
    unify_bing_naming_scheme()