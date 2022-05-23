"""
Train an auto-encoder using your data.
The pre-trained model will be saved into self_discovery/Checkpoints/Autoencoder/ directory.
"""

import argparse
import os
import random
import sys
from datetime import datetime
from glob import glob

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn.utils import shuffle

from unet3d import *

sys.setrecursionlimit(40000)  # prevent any program from getting into infinite recursion

parser = argparse.ArgumentParser()

parser.add_argument("--arch", default="Unet", type=str, help="Vnet|Unet")
parser.add_argument("--backbone", default="", type=str)
parser.add_argument("--decoder", dest="decoder_block_type", help="transpose | upsampling",
                    default="upsampling", type=str)

parser.add_argument("--input_rows", default=128, type=int, help="input rows")
parser.add_argument("--input_cols", default=128, type=int, help="input cols")
parser.add_argument("--input_deps", default=64, type=int, help="input deps")

parser.add_argument("--verbose", default=1, type=int, help="verbose")
parser.add_argument("--weights", dest="weights", default=None, type=str, help="pre-trained weights")

parser.add_argument("--batch_size", default=8, type=int, help="batch size")
parser.add_argument("--seed", default=1, type=int)

parser.add_argument("--data_dir", dest="data_dir", default=None, help="path to data")
parser.add_argument("--model_path", dest="model_path", default="Checkpoints/Autoencoder", help="path to save model")

args = parser.parse_args()

assert args.data_dir is not None

random.seed(args.seed)

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)


def date_str():
    return datetime.now().__str__().replace("-", "_").replace(" ", "_").replace(":", "_")


class SetupConfig:
    nb_epoch = 10000
    patience = 50
    lr = 1e-3

    def __init__(self,
                 model=args.arch,
                 backbone=args.backbone,
                 input_rows=args.input_rows,
                 input_cols=args.input_cols,
                 input_deps=args.input_deps,
                 batch_size=args.batch_size,
                 decoder_block_type=args.decoder_block_type,
                 nb_class=1,
                 # model_path=args.model_path,
                 verbose=args.verbose):

        self.model = model
        self.backbone = backbone

        # experiment name
        self.exp_name = model + "_autoencoder"

        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps

        self.batch_size = batch_size
        self.verbose = verbose
        self.decoder_block_type = decoder_block_type
        self.nb_class = nb_class

        # self.model_path = model_path

        if nb_class > 1:
            self.activation = "softmax"
        else:
            self.activation = "sigmoid"

    def display(self):
        """Display Configuration values.
        if want to call this function, please use:

        a = setup_config()
        print(a.display())

        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print(f"{a:30} {getattr(self, a)}")
        print("\n")


class DataGenerator(tf.keras.utils.Sequence):
    """
    Every Sequence must implement the __getitem__ and the __len__ methods.
    If you want to modify your dataset between epochs you may implement on_epoch_end.  --> for instance, shuffle
    The method __getitem__ should return a complete batch.

    Sequence are a safer way to do multiprocessing. This structure guarantees that
    the network will only train once on each sample per epoch which is not the case with generators.
    """

    def __init__(self,
                 directory,
                 batch_size=16,
                 dim=(128, 128, 64)
                 ):

        self.directory = directory
        self.image_paths = self.get_list_of_images(directory)
        self.batch_size = batch_size
        self.dim = dim

    def get_list_of_images(self, path):
        """ glob all the images and return target images in a list """
        try:
            images = glob(os.path.join(path, "*"))
            return images
        except FileNotFoundError:
            print("Wrong file or file path")

    def data_loader(self, file_list):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        input_depth = self.dim[2]

        x = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")
        y = np.zeros((self.batch_size, 1, input_rows, input_cols, input_depth), dtype="float")

        count = 0

        # recursively load data
        for i, file in enumerate(file_list):
            # # pay attention to np.load (how to deal with nifti/dicom file?)
            img = np.load(file)

            img = resize(image=img,  # date type: ndarray
                         output_shape=(input_rows, input_cols, input_depth),
                         preserve_range=True  # Whether to keep the original range of values. Otherwise, the input
                         # image is converted according to the conventions of `img_as_float`.
                         )
            img = np.expand_dims(img, axis=0)  # (128,128,64) --> (1,128,128,64)

            # # x and y are the same ???  one for process and another for ground truth?
            x[count, :, :, :] = img
            y[count, :, :, :] = img

            count += 1

        x, y = shuffle(x, y, random_state=0)
        return x, y

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))  # 7.6 epoch --> 8 epoch to load all the data

    def __getitem__(self, index):

        file_names = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        return self.data_loader(file_names)

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        Shuffle all images after an epoch (API in tf.keras.utils.Sequence)"""
        np.random.shuffle(self.image_paths)


if __name__ == "__main__":
    config = SetupConfig()
    config.display()

    # # arch need to be filled
    if args.arch == "Vnet":
        pass
    elif args.arch == "Unet":
        model = unet_model_3d(input_shape=(1, config.input_rows, config.input_cols, config.input_deps),
                              batch_normalization=True)

    if args.weights is not None:
        print(f"Load the pre-trained weights from {args.weights}")
        model.load_weights(filepath=args.weights)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
                  loss="MSE",
                  metrics=["MAE", "MSE"])

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=config.patience,
                                                      verbose=0,
                                                      mode='min')

    check_point = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_path, config.exp_name + ".h5"),
                                                     monitor='val_loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='min')

    # Reduce learning rate when a metric has stopped improving
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                        factor=.5,
                                                        patience=20,
                                                        min_delta=.0001,
                                                        min_lr=1e-6,
                                                        verbose=1)

    callbacks = [check_point, early_stopping, lr_scheduler]

    # training_generator = DataGenerator(directory=os.path.join(args.data_dir, 'train/'),
    #                                    batch_size=args.batch_size,
    #                                    dim=(config.input_rows, config.input_cols, config.input_deps))
    #
    # validation_generator = DataGenerator(directory=os.path.join(args.data_dir, 'validation/'),
    #                                      batch_size=args.batch_size,
    #                                      dim=(config.input_rows, config.input_cols, config.input_deps))
    #
    # model.fit(x=training_generator,
    #           validation_data=validation_generator,
    #           steps_per_epoch=len(training_generator) // args.batch_size,
    #           validation_steps=len(validation_generator) // args.batch_size,
    #           epochs=config.nb_epoch,
    #           max_queue_size=20,
    #           workers=7,
    #           use_multiprocessing=True,
    #           shuffle=True,
    #           verbose=config.verbose,
    #           callbacks=callbacks)
