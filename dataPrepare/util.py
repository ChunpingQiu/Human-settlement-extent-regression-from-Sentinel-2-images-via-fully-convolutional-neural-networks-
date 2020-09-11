# @Date:   2019-12-20T15:46:36+01:00
# @Last modified time: 2019-12-21T16:51:35+01:00



from keras import backend as K
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

from tensorflow.python.ops import nn

import numpy as np
#################################################################
def masked_loss_function(y_true, y_pred):
    """ cumpute loss ignoring labels according to y_true[none, :, :, 0] where 17 means no data area
    # Arguments
    # Returns
    """
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

    mask = K.cast(tf.not_equal(y_true, 17), K.floatx())

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

#     # scale preds so that the class probas of each sample sum to 1
#     y_pred = y_pred / math_ops.reduce_sum(y_pred, -1, True)
# # manual computation of crossentropy
    epsilon_ = ops.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)


    return -tf.reduce_sum(tf.divide(math_ops.reduce_sum(y_true * math_ops.log(y_pred), -1), tf.reduce_sum(mask)) )
    #return math_ops.reduce_sum(y_true * math_ops.log(y_pred), -1)

def binary_crossentropy_function(y_true, y_pred):
    """ cumpute loss  to be consistent when weight is learned
    # Arguments
    # Returns
    """

    # print(y_true.shape, y_pred.shape)
    #
    # print(tf.keras.metrics.binary_crossentropy(y_true, y_pred).shape)
    # print(K.mean( tf.keras.metrics.binary_crossentropy(y_true, y_pred) ).shape)

    return K.mean( tf.keras.metrics.binary_crossentropy(y_true, y_pred) )

def binary_crossentropy_function_weightSample(y_true, y_pred):
    """ cumpute loss  to be consistent when weight is learned
    # Arguments
    # Returns
    """

    print(y_true.shape, y_pred.shape)

    sampleWeight=math_ops.reduce_max(y_true, axis=-1)
    y_true=K.argmax(y_true, axis=-1)
    y_true=K.one_hot(y_true, 2)

    print(y_true.shape, y_pred.shape, sampleWeight.shape)

    # y_pred = K.reshape(y_pred, (-1, 1))
    # y_true = K.reshape(y_true, (-1, 1))
    # sampleWeight = K.reshape(sampleWeight, (-1, 1))

    y_true=K.cast(y_true, K.floatx())
    sampleWeight=K.cast(sampleWeight, K.floatx())

    print(tf.keras.metrics.binary_crossentropy(y_true, y_pred).shape)
    #print(K.mean( tf.keras.metrics.binary_crossentropy(y_true, y_pred) ).shape)

    return K.mean( sampleWeight*tf.keras.metrics.binary_crossentropy(y_true, y_pred) )


def binary_crossentropy_function_(y_true, y_pred):
    """ cumpute loss  to be consistent when weight is learned
    # Arguments
    # Returns
    """

    print(y_true.shape, y_pred.shape)

    #sampleWeight=math_ops.reduce_max(y_true, axis=-1)
    y_true=K.argmax(y_true, axis=-1)
    y_true=K.one_hot(y_true, 2)

    #print(y_true.shape, y_pred.shape, sampleWeight.shape)

    # y_pred = K.reshape(y_pred, (-1, 1))
    # y_true = K.reshape(y_true, (-1, 1))
    # sampleWeight = K.reshape(sampleWeight, (-1, 1))

    y_true=K.cast(y_true, K.floatx())
    #sampleWeight=K.cast(sampleWeight, K.floatx())

    print(tf.keras.metrics.binary_crossentropy(y_true, y_pred).shape)
    #print(K.mean( tf.keras.metrics.binary_crossentropy(y_true, y_pred) ).shape)

    return K.mean(tf.keras.metrics.binary_crossentropy(y_true, y_pred) )

################################################################################
################################################################################
def recall_m(y_true_oneHot, y_pred_oneHot):
        y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
        y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

        'because class 0 is the target class'
        y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
        y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())

# def recall_m(y_true, y_pred):
#         y_pred=K.round(y_pred)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true_oneHot, y_pred_oneHot):

        y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
        y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

        'because class 0 is the target class'
        y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
        y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())
# def precision_m(y_true, y_pred):
#
#         y_pred=K.round(y_pred)

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def dice_coef(y_true_oneHot, y_pred_oneHot):

    y_true=K.cast(K.argmax(y_true_oneHot, axis=-1), K.floatx())
    y_pred=K.cast(K.argmax(y_pred_oneHot, axis=-1), K.floatx())

    'because class 0 is the target class'
    y_true = K.cast(math_ops.subtract(1.0, y_true), K.floatx())#
    y_pred = K.cast(math_ops.subtract(1.0, y_pred), K.floatx())

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    # print(y_true_f.shape, y_pred_f.shape)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)#, _val_f1, _val_recall, _val_precision
#
# def dice_coef_loss(y_true, y_pred):
# 	return -dice_coef(y_true, y_pred)


#########################################

def _to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.

  Arguments:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  x = ops.convert_to_tensor(x)
  if x.dtype != dtype:
    x = math_ops.cast(x, dtype)
  return x

def binary_crossentropy_(target, output, from_logits=False):
  """Binary crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  """
  # Note: nn.softmax_cross_entropy_with_logits
  # expects logits, Keras expects probabilities.
  if not from_logits:
    # transform back to logits
    epsilon_ = _to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output / (1 - output))
  return nn.weighted_cross_entropy_with_logits(target, output, pos_weight=5)

def binary_crossentropy_weight(y_true, y_pred):
  return K.mean(binary_crossentropy_(y_true, y_pred), axis=-1)

#########################################
# def weighted_b_cross_entropy(_w):
#     def weighted_b_cross_entropy_core(y_true, y_pred):
#         """
#         """
#         # hyper param
#         print(_w)
#         y_pred = K.clip(y_pred, K.epsilon(), 1)
#
#         # _loss = -y_true*K.log(y_pred)*_w + -(1-y_true)*K.log(1-y_pred)
#         # return _loss
#
#       # Compute cross entropy from probabilities.
#         bce = y_true * math_ops.log(y_pred + K.epsilon())*_w
#         bce += (1 - y_true) * math_ops.log(1 - y_pred + K.epsilon())
#         return -bce
#
#
#     return weighted_b_cross_entropy_core
