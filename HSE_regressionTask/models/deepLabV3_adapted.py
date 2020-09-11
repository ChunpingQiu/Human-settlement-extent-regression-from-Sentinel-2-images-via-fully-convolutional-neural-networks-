# @Date:   2020-02-22T16:38:20+01:00
# @Last modified time: 2020-04-18T17:11:54+02:00



# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on:
https://github.com/bonlime/keras-deeplab-v3-plus


# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input, multiply, Average
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
import cbam
from keras.initializers import Constant
from keras import backend as K
import sys

import tensorflow as tf



class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v

#
# def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
#     in_channels = inputs._keras_shape[-1]
#     pointwise_conv_filters = int(filters * alpha)
#     pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
#     x = inputs
#     prefix = 'expanded_conv_{}_'.format(block_id)
#     if block_id:
#         # Expand
#
#         x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
#                    use_bias=False, activation=None,
#                    name=prefix + 'expand')(x)
#         x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                                name=prefix + 'expand_BN')(x)
#         x = Activation(relu6, name=prefix + 'expand_relu')(x)
#     else:
#         prefix = 'expanded_conv_'
#     # Depthwise
#     x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
#                         use_bias=False, padding='same', dilation_rate=(rate, rate),
#                         name=prefix + 'depthwise')(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                            name=prefix + 'depthwise_BN')(x)
#
#     x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
#
#     # Project
#     x = Conv2D(pointwise_filters,
#                kernel_size=1, padding='same', use_bias=False, activation=None,
#                name=prefix + 'project')(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                            name=prefix + 'project_BN')(x)
#
#     if skip_connection:
#         return Add(name=prefix + 'add')([inputs, x])
#
#     # if in_channels == pointwise_filters and stride == 1:
#     #    return Add(name='res_connect_' + str(block_id))([inputs, x])
#
#     return x


def deepLabV3_out(img_input, classes=1, out_shape=64, atrous_rates = (1, 2, 4)):
    """ Instantiates the Deeplabv3+ architecture

    # Arguments
        input_tensor: Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        atrous_rates: rates for the aspp
        out_shape: shape of output image.
        classes: number of desired classes.


    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    inc_rate=2
    dim=16
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%para: ')
    print('inc_rate: ' ,inc_rate)
    print('dim: ' ,dim)
    # print('finalFeatureNum (fixed in other three funcitions): ' ,finalFeatureNum)

    x = Conv2D(dim, (3, 3), strides=1,#, strides=(2, 2),
               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    dim=dim*inc_rate
    x = Conv2D(dim, (3, 3), strides=1,#, strides=(2, 2),
               name='entry_flow_conv1_2', use_bias=False, padding='same')(x)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)

    dim=dim*inc_rate
    x = _xception_block(x, [dim, dim, dim], 'entry_flow_block', skip_connection_type='conv', stride=2, depth_activation=False)
    skip1 = x ##size 1/2

    for i in range(2):
        x = _xception_block(x, [dim, dim, dim], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=1,
                            depth_activation=False)


    # x = _xception_block(x, [dim, dim, dim], 'exit_flow_block1',
    #                     skip_connection_type='conv', stride=1, rate=1,
    #                     depth_activation=False)


    # simple 1x1
    b0 = Conv2D(int(dim/4), (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    #atrous_rates 3+(x-1)x2=3,5,9
    b1 = SepConv_BN(x, int(dim/4), 'aspp1',
                    rate=1, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, int(dim/4), 'aspp2',
                    rate=2, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, int(dim/4), 'aspp3',
                    rate=4, depth_activation=True, epsilon=1e-5)
    x0 = Concatenate()([b0, b1, b2, b3])

    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    x = Dropout(0.1)(x0)

    x = Concatenate()([x, dec_skip1])

    o = Conv2D(classes, (1, 1), padding='same', activation = 'sigmoid', name="hse")(x)
    if out_shape != x.shape[1]:
        print('BilinearUpsampling needed:')
        o= BilinearUpsampling(output_size=(out_shape, out_shape))(o)

    return o



def dl_net(input_size = (128,128,10)):

    inputs = Input(input_size)

    o0 = deepLabV3_out(inputs)
    model = Model(inputs, o0, name='hseNN')

    return model
