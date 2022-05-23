"""
Extract 3D visual words from train and validation images.
The data and their labels will be saved into self_discovery/TransVW_data directory.
"""

import argparse
import os
import sys
from glob import glob
import random

import numpy as np

sys.setrecursionlimit(40000)

parser = argparse.ArgumentParser()

parser.add_argument("--input_rows", default=64, type=int, help="input rows")
parser.add_argument("--input_cols", default=64, type=int, help="input cols")
parser.add_argument("--input_deps", default=32, type=int, help="input deps")

parser.add_argument("--crop_rows", default=64, type=int, help="crop rows")
parser.add_argument("--crop_cols", default=64, type=int, help="crop cols")

parser.add_argument("--data_dir", default=None, type=str, help="path to pre-trained dataset")
parser.add_argument("--save", default="./TransVW_data", type=str, help="path to save processed 3D cubes")

parser.add_argument("--is_normalized", default=False, action="store_true",
                    help="True if the images are already normalized")
parser.add_argument("--nb_instances", default=200, type=int, help="# of instances of each pattern")
parser.add_argument("--nb_instances_val", default=30, type=int, help="# of instances of each pattern for validation")
parser.add_argument("--nb_classes", default=200, type=int, help="# of instances of each pattern")

parser.add_argument("--minPatchSize", default=50, type=int, help="minimum cube size")
parser.add_argument("--maxPatchSize", default=80, type=int, help="maximum cube size")

parser.add_argument("--distance_threshold", default=5, type=int, help="minimum distance between points")
parser.add_argument("--prev_coordinates", default=None, help="address of previously generated coordinates")

parser.add_argument("--train_features", default="val_features", help="path to the saved features for train images")
parser.add_argument("--val_features", default="val_features", help="path to the saved features for validation images")

args = parser.parse_args()

seed = 1
random.seed(args.seed)

assert args.data_dir is not None
assert args.save is not None
assert args.train_features is not None
assert args.val_features is not None

if not os.path.exists(args.save):
    os.makedirs(args.save)

# -------load train features and validation features after the processing of feature_extractor.py--------
train_features = np.load(args.train_features)
print("train features:", train_features.shape)

validation_features = np.load(args.val_features)
print("validation features:", validation_features.shape)

target_path_train = os.path.join(args.data_dir, "train")
target_path_val = os.path.join(args.data_dir, "validation")

""" Pay attention to that the file need to be *.npy which can be loaded successfully"""
train_image_list = glob(os.path.join(target_path_train, "*.npy"))
train_image_list.sort()

val_image_list = glob(os.path.join(target_path_val, "*.npy"))
val_image_list.sort()


class setup_config:
    hu_max = 1000.0
    hu_min = -1000.0
    HU_thred = (-150.0 - hu_min) / (hu_max - hu_min)

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,

                 # #
                 train_fold=[0, 1, 2, 3, 4],
                 valid_fold=[5, 6],
                 test_fold=[7, 8, 9],

                 len_depth=None,
                 lung_min=0.7,
                 lung_max=1.0,
                 is_normalized=False,
                 minPatchSize=50,
                 maxPatchSize=100,
                 multi_res=True,
                 nb_instances=200,
                 nb_instances_val=30,
                 nb_classes=200,
                 save="./"):

        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps

        self.crop_rows = crop_rows
        self.crop_cols = crop_cols

        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale

        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold
        self.len_depth = len_depth

        self.lung_min = lung_min
        self.lung_max = lung_max

        self.is_normalized = is_normalized

        self.minPatchSize = minPatchSize
        self.maxPatchSize = maxPatchSize
        self.multi_res = multi_res

        self.nb_instances = nb_instances
        self.nb_instances_val = nb_instances_val
        self.nb_classes = nb_classes

        self.save = save

    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print(f"{a:30} {getattr(self, a)}")
        print("\n")


config = setup_config(input_rows=args.input_rows,
                      input_cols=args.input_cols,
                      input_deps=args.input_deps,
                      crop_rows=args.crop_rows,
                      crop_cols=args.crop_cols,
                      len_border=100,
                      len_border_z=30,
                      len_depth=3,
                      lung_min=0.7,
                      lung_max=0.15,
                      DATA_DIR=args.data_dir,
                      is_normalized=args.is_normalized,
                      minPatchSize=args.minPatchSize,
                      maxPatchSize=args.maxPatchSize,
                      multi_res=args.multi_res,
                      nb_instances=args.nb_instances,
                      nb_instances_val=args.nb_instances_val,
                      nb_classes=args.nb_classes,
                      save=args.save,
                      )
config.display()


def initialization():
    coordinates = []
    ref_selected = np.zeros((1, train_features.shape[0]))
    visited_labels = []

    if args.prev_coordinates is not None:  # address of previously generated coordinates
        pass  # #


def get_random_coordinate(config, img_array, coordinates):
    size_x, size_y, size_z = img_array.shape

    # #
    if size_z - config.input_deps - config.len_depth - 1 - config.len_border_z < config.len_border_z:
        return None































