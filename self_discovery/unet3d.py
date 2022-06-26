import tensorflow as tf

tf.keras.backend.set_image_data_format("channels_first")


def create_convolution_block(input_layer,
                             n_filters,  # determine the output channels
                             batch_normalization=False,
                             kernel=(3, 3, 3),
                             activation=None,
                             padding='same',  # keep the output size is the same as the input size
                             strides=(1, 1, 1),
                             layer_depth=None  # in order to record the index of level (list type)
                             ):
    """ Returns a combination of Conv3D + Batch Normalization + relu """

    layer = tf.keras.layers.Conv3D(filters=n_filters,
                                   kernel_size=kernel,
                                   padding=padding,
                                   strides=strides,
                                   name="depth_" + str(layer_depth) + "_conv")(input_layer)

    if batch_normalization is not None:
        layer = tf.keras.layers.BatchNormalization(axis=1,
                                                   name="depth_" + str(layer_depth) + "_bn")(layer)

    if activation is None:
        return tf.keras.layers.Activation('relu',
                                          name='depth_' + str(layer_depth) + '_relu')(layer)
    else:
        return activation()(layer)


def get_up_convolution(n_filters,
                       pool_size,
                       kernel_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution is not None:
        return tf.keras.layers.Conv3DTranspose(filters=n_filters,
                                               kernel_size=kernel_size,
                                               strides=strides)

    else:
        # Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1] and size[2] respectively.
        return tf.keras.layers.UpSampling3D(size=pool_size)


def unet_model_3d(input_shape,
                  pool_size=(2, 2, 2),
                  n_labels=1,
                  deconvolution=False,
                  depth=4,
                  n_base_filters=32,
                  batch_normalization=False,
                  activation_name="sigmoid"):
    """
    input_shape --> (n_channels, x_size, y_size, z_size)  (because of  K.set_image_data_format("channels_first"))
        pay attention that x, y, and z sizes must be
        divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    pool_size --> Pool size for the max pooling operation
    n_labels --> # of binary labels that the model is learning
    """

    inputs = tf.keras.Input(input_shape)
    current_layer = inputs

    # store the nn architecture
    levels = list()

    num_layer = 0

    # ----------- add levels with max pooling ------------
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (2 ** layer_depth),
                                          batch_normalization=batch_normalization,
                                          layer_depth=num_layer)
        num_layer += 1

        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization,
                                          layer_depth=num_layer)
        num_layer += 1

        if layer_depth < depth - 1:
            current_layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])  # Conv_Block1 + Conv_Block2 + MaxPooling
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # ----------- add levels with up-convolution or up-sampling ------------
    # reversely recursion
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=current_layer.shape[1])(current_layer)  # # _keras_shape -> shape

        concat = tf.keras.layers.concatenate([up_convolution, levels[layer_depth][1]], axis=1)

        current_layer = create_convolution_block(input_layer=concat,
                                                 n_filters=levels[layer_depth][1].shape[1],
                                                 batch_normalization=batch_normalization,
                                                 layer_depth=num_layer)
        num_layer += 1

        current_layer = create_convolution_block(input_layer=current_layer,
                                                 n_filters=levels[layer_depth][1].shape[1],
                                                 batch_normalization=batch_normalization,
                                                 layer_depth=num_layer)
        num_layer += 1

    # 1 by 1 conv to change # of channels
    final_convolution = tf.keras.layers.Conv3D(filters=n_labels,
                                               kernel_size=(1, 1, 1),
                                               name="final_conv")(current_layer)

    act = tf.keras.layers.Activation(activation=activation_name)(final_convolution)

    model = tf.keras.Model(inputs=inputs,
                           outputs=act)

    return model  # untrained 3D Unet model


def compute_level_output_shape(n_filters,
                               depth,
                               pool_size,
                               image_shape):
    pass


if __name__ == '__main__':
    model = unet_model_3d(input_shape=(1, 64, 64, 32),
                          batch_normalization=True)

    model.summary()
