"""
Extract and save the deep features of each patient in the dataset using the pre-trained auto-encoder
"""

import sys
import argparse
from glob import glob
import os

import numpy as np
import tensorflow as tf
from skimage.transform import resize

from unet3d import *

sys.setrecursionlimit(40000)

parser = argparse.ArgumentParser()

parser.add_argument("--arch", default="Unet", type=str, help="Vnet|Unet")

parser.add_argument("--input_rows", default=128, type=int, help="input rows")
parser.add_argument("--input_cols", default=128, type=int, help="input cols")
parser.add_argument("--input_deps", default=64, type=int, help="input deps")

parser.add_argument("--verbose", default=1, type=int, help="verbose")
parser.add_argument("--weights", dest="weights", default=None, type=str, help="pre-trained weights")

parser.add_argument("--batch_size", default=8, type=int, help="batch size")

parser.add_argument("--learning_rate", default=.001, type=float, help="learning rate")

parser.add_argument("--data_dir", dest="data_dir", default=None, help="path to images")

args = parser.parse_args()

assert args.data_dir is not None
assert args.weights is not None

input_rows = args.input_rows
input_cols = args.input_cols
input_deps = args.input_deps

# # arch need to be filled
if args.arch == "Vnet":
    pass
elif args.arch == "Unet":
    model = unet_model_3d(input_shape=(1, args.input_rows, args.input_cols, args.input_deps),
                          batch_normalization=True)

model.load_weights(args.weights)  # self_discovery/Checkpoints/Autoencoder/Unet_autoencoder.h5

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                                momentum=.9,
                                                decay=.0,
                                                nesterov=False,
                                                clipnorm=1),
              loss="MSE",
              metrics=["MAE", "MSE"])

# ------- extract the encoder for feature extraction ------------
x = model.get_layer('depth_7_relu').output
x = tf.keras.layers.GlobalAveragePooling3D()(x)

encoder_model = tf.keras.Model(inputs=model.input, outputs=x)
encoder_model.summary()

# ---------------------- train_images_feature_extractor --------------------
train_images_list = glob(os.path.join(args.data_dir, "train", "*"))  # dataset/train/*
train_images_list.sort()

train_features = np.zeros((len(train_images_list), 512), dtype=np.float32)
count = 0

for image in train_images_list:
    """ image should be loaded by numpy """
    img = np.load(image)
    img = resize(image=img,
                 output_shape=(input_rows, input_cols, input_deps),
                 preserve_range=True)
    img = np.expand_dims(img, axis=0)

    x = np.zeros(shape=(1, 1, input_rows, input_cols, input_deps),
                 dtype=np.float)
    x[0, :, :, :, :] = img

    # distribute pseudo labels
    feature = encoder_model.predict(x)
    train_features[count, :] = feature
    count += 1
    print(count)  # # why need this expression?

np.save("train_features", train_features)
print("Train features has been saved.")

# ---------------------- val_images_feature_extractor --------------------
val_images_list = glob(os.path.join(args.data_dir, "validation", "*"))  # dataset/validation/*
val_images_list.sort()

val_features = np.zeros((len(val_images_list), 512), dtype=np.float32)
count = 0

for image in val_images_list:
    x = np.zeros(shape=(1, 1, input_rows, input_cols, input_deps),
                 dtype=np.float)
    img = np.load(image)
    img = resize(image=img,
                 output_shape=(input_rows, input_cols, input_deps),
                 preserve_range=True)
    img = np.expand_dims(img, axis=0)

    x[0, :, :, :, :] = img
    feature = encoder_model.predict(x)
    val_features[count, :] = feature
    count += 1
    print(count)

np.save("validation_features", val_features)
print("Validation features has been saved.")
