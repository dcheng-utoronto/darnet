import configparser
import os
import inspect

config = configparser.ConfigParser(
    interpolation=configparser.ExtendedInterpolation())
config_file = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "config.ini")
config.read(config_file)
