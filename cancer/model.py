import os.path

from keras.layers import (
    Dropout, Dense, BatchNormalization, Activation, InputLayer,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D
)
from keras.models import Sequential

from . import common


def conv2d(filters, **kwargs):
    return Conv2D(
        filters, kernel_size=(3, 3), strides=(1, 1),
        padding='same', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        **kwargs
    )


def relu():
    return Activation('relu')


def maxpool2d():
    return MaxPooling2D(pool_size=(2, 2), padding='same')


def dropout(rate):
    return Dropout(rate)


def batch_normalization():
    return BatchNormalization()


def add_conv_segment(model, filters, dropout_rate):
    model.add(conv2d(filters))
    model.add(batch_normalization())
    model.add(relu())
    model.add(maxpool2d())
    model.add(dropout(dropout_rate))


def add_dense_segment(model, width, dropout_rate):
    model.add(Dense(width))
    model.add(batch_normalization())
    model.add(relu())
    model.add(dropout(dropout_rate))


def get_weights_file_path(input_x, input_y):
    return os.path.join(common.TOP, f'weights.best.{input_x}x{input_y}.hdf5')


def get_compiled_model(input_x, input_y, dropout_rate=0.0):
    outputs = len(common.CLASSES)
    pretrained_model = common.get_pretrained_model(input_x, input_y)
    model = Sequential()
    model.add(InputLayer(input_shape=pretrained_model.output_shape[1:]))
    pretrained_channels = pretrained_model.output_shape[-1]

    add_conv_segment(model, pretrained_channels * 2, dropout_rate)

    model.add(GlobalAveragePooling2D())

    add_dense_segment(model, pretrained_channels, dropout_rate)
    add_dense_segment(model, int(pretrained_channels / outputs), dropout_rate)

    model.add(Dense(outputs, activation='softmax'))
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
