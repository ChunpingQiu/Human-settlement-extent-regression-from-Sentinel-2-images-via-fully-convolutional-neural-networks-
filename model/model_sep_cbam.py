# @Date:   2020-02-05T15:27:26+01:00
# @Last modified time: 2020-04-19T13:54:43+02:00

from keras.models import *
from keras.layers import *

import tensorflow as tf
from keras.initializers import Constant


import cbam

"Custom loss layer"
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        # for i in range(self.nb_outputs):
        #     self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
        #                                       initializer=Constant(0.), trainable=True)]##-1.5363 -5.6260

        "initial the weight-related parameters"
        self.log_vars += [self.add_weight(name='log_var' + str(0), shape=(1,),
                                          initializer=Constant(0), trainable=True)]
        self.log_vars += [self.add_weight(name='log_var' + str(1), shape=(1,),
                                          initializer=Constant(0), trainable=True)]

        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        #loss = 0
        # for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
        #     precision = K.exp(-log_var[0])
        #     loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        # precision1 = K.exp(-self.log_vars[0][0])

        'for regression loss'
        precision1 = K.exp(-self.log_vars[0][0])/2.0

        #print(ys_true[0].shape, ys_pred[0].shape, ys_true[1].shape, ys_pred[1].shape)
        # loss1 = K.mean( tf.keras.metrics.mean_absolute_error(ys_true[0], ys_pred[0]) )
        # loss1 = K.mean( tf.keras.metrics.sparse_categorical_crossentropy(ys_true[0], ys_pred[0]) )
        'change the loss accordingly when necessary'
        loss1 = K.mean( mean_absolute_error_weight(ys_true[0], ys_pred[0]) )

        loss = K.sum( precision1*loss1 + self.log_vars[0][0]/2.0)
        precision2 = K.exp(-self.log_vars[1][0])

        'for categorical_crossentropy loss'
        'change the loss accordingly'
        loss2 =  K.mean( tf.keras.metrics.categorical_crossentropy(ys_true[1], ys_pred[1]) )
        # loss2 =  K.mean( tf.keras.metrics.mean_absolute_error(ys_true[1], ys_pred[1]) )

        loss = loss + K.sum( precision2*loss2 + self.log_vars[1][0]/2.0)

        return loss, loss1, loss2, self.log_vars[0][0], self.log_vars[1][0]#K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss, loss1, loss2, var1, var2 = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss)
        # We won't actually use the output.

        out = tf.convert_to_tensor(tf.stack([loss1, loss2, var1, var2], -1))
        print(out)
        return out

    def compute_output_shape(self, inputs):
        return (4, )


def hseNet_0(inputs, lay_per_block=4, inc_rate = 2, bn=1, dim=16):

    'h*w'
    conv0 = Conv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(inputs)

    if bn==1:
        print('with BN')
        conv0 = BatchNormalization(axis=-1)(conv0)
    conv0 = Activation('relu')(conv0)

    #############################################
    for i in np.arange(0, lay_per_block/2-1):
        # print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv0 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
        if bn==1:
            print('with BN')
            conv0 = BatchNormalization(axis=-1)(conv0)
        conv0 = Activation('relu')(conv0)

    'increase width'
    #############################################
    'similar to that in hse_isprs to be comparable; to have a not so wide nn; later pooling'
    dim=dim*inc_rate
    for i in np.arange(0, lay_per_block/2):
        # print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv0 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv0)
        if bn==1:
            print('with BN')
            conv0 = BatchNormalization(axis=-1)(conv0)
        conv0 = Activation('relu')(conv0)

    #"how to pooling?!"
    #############################################

    #"original idea"
    pool0 = MaxPooling2D((2, 2))(conv0)
    pool1 = AveragePooling2D(pool_size=2)(conv0)
    merge0 = Concatenate()([pool0,pool1])
    #############################################


    'h*w /2'
    'similar to hse_isprs'
    dim=dim*inc_rate*2
    conv1 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge0)
    if bn==1:
        print('with BN')
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = Activation('relu')(conv1)

    for i in np.arange(0, lay_per_block/2-1):
        # print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv1 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
        if bn==1:
            print('with BN')
            conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)

    'increase width'
    #############################################
    dim=dim*inc_rate
    for i in np.arange(0, lay_per_block/2):
        # print(str(i) +'in' +str(lay_per_block-1), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        conv1 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(conv1)
        if bn==1:
            print('with BN')
            conv1 = BatchNormalization(axis=-1)(conv1)
        conv1 = Activation('relu')(conv1)

    return merge0, conv1


def sen2_LCZ_HSE(inputs, num_classes=(1,17), output=(64,16), bn=1, dropRate=0.1, fusion=1, flow=2, taskAttation=0, shareNet=0, lczFeatureReturn=0):

    initialSize=int(inputs.shape[1])

    if shareNet==0:
        merge0, conv1=hseNet_0(inputs, bn=bn)

    if fusion==1:
        'prediction'
        outputs_32 = Conv2D(num_classes[1], (1, 1), activation='softmax', kernel_initializer='he_normal')(merge0)
        _up_rate=int(initialSize/int(outputs_32.shape[1]))
        outputs_32=UpSampling2D(size=(_up_rate, _up_rate))(outputs_32)

    #############################################
    if taskAttation==1:
        x0 = cbam.attach_attention_module(conv1,'cbam_block')
        x1 = cbam.attach_attention_module(conv1,'cbam_block')

    #############################################
    if taskAttation>0:
        drop_0 = Dropout(dropRate)(x0)
    else:
        drop_0 = Dropout(dropRate)(conv1)
    o_0 = Conv2D(num_classes[0], (1, 1), padding='same', activation = 'sigmoid', name="hse")(drop_0)
    #############################################

    'there is only one task after here, and only one branch'
    if taskAttation>0:
        conv1_=x1
    else:
        conv1_=conv1

    if fusion==1:
        'final prediction'
        outputs_16, outputs_8, outputs, conv3, conv2 = LCZNet(conv1_, initialSize,num_classes=num_classes[1], fusion=fusion, dropRate=dropRate, bn=bn)
        # o=Average(name="lcz")([outputs, outputs_32, outputs_16, outputs_8])#
    else:
        o, conv3, conv2=LCZNet(conv1_,initialSize,num_classes=num_classes[1], fusion=fusion, dropRate=dropRate, bn=bn)


    if flow ==2:

        if lczFeatureReturn==0:
            if fusion ==1:
                return [o_0, Average(name="lcz")([outputs, outputs_32, outputs_16, outputs_8])]
            else:
                return [o_0, o]#

        else:

            if fusion ==1:
                return [o_0, outputs, outputs_32, outputs_16, outputs_8, conv3, conv2, conv1]#return [o_0, o, conv1]
            else:
                return [o_0, o, conv3, conv2, conv1]#

    if flow ==1:
        if fusion ==1:
            return Average(name="lcz")([outputs, outputs_32, outputs_16, outputs_8])
        else:
            return o#


    if flow ==0:
        return o_0


def LCZNet(conv1, initialSize, num_classes=17, inc_rate = 2, fusion=1, dropRate=0.1, bn=1):
    "how to pooling?!"

    dim=int(conv1.shape[-1])
    #############################################
    "original idea"
    pool0 = MaxPooling2D((2, 2))(conv1)
    pool1 = AveragePooling2D(pool_size=2)(conv1)
    merge1 = Concatenate()([pool0, pool1])
    #############################################
    if fusion==1:
        'prediction'
        # _pool_size=int(int(merge1.shape[1])/output[1])
        # x = AveragePooling2D(pool_size=_pool_size)(merge1)#Flatten
        # print(x.shape)
        outputs_16 = Conv2D(num_classes, (1, 1), activation='softmax', kernel_initializer='he_normal')(merge1)
        _up_rate=int(initialSize/int(outputs_16.shape[1]))
        outputs_16=UpSampling2D(size=(_up_rate, _up_rate))(outputs_16)

    'dropOut'
    merge1 = Dropout(dropRate)(merge1)

    'h*w /4'
    dim=dim*inc_rate
    conv2 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge1)
    if bn==1:
        print('with BN')
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Activation('relu')(conv2)

    "how to pooling?!"
    #############################################

    "original idea"
    pool0 = MaxPooling2D((2, 2))(conv2)
    pool1 = AveragePooling2D(pool_size=2)(conv2)
    merge2 = Concatenate()([pool0,pool1])
    #############################################

    if fusion==1:
        'prediction'
        # _pool_size=int(int(merge2.shape[1])/output[1])
        # x = AveragePooling2D(pool_size=_pool_size)(merge2)
        # print(x.shape)

        outputs_8 = Conv2D(num_classes, (1, 1), activation='softmax', kernel_initializer='he_normal')(merge2)
        _up_rate=int(initialSize/int(outputs_8.shape[1]))
        outputs_8=UpSampling2D(size=(_up_rate, _up_rate))(outputs_8)

    'dropOut'
    merge2 = Dropout(dropRate)(merge2)

    'h*w /8'
    dim=dim*inc_rate
    conv3 = SeparableConv2D(dim, (3, 3), padding='same', kernel_initializer = 'he_normal')(merge2)
    if bn==1:
        print('with BN')
        conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Activation('relu')(conv3)

    'prediction'
    outputs = Conv2D(num_classes, (1, 1), activation='softmax', kernel_initializer='he_normal')(conv3)
    _up_rate=int(initialSize/int(outputs.shape[1]))

    if fusion==1:
        outputs=UpSampling2D(size=(_up_rate, _up_rate))(outputs)
        return outputs_16, outputs_8, outputs, conv3, conv2
    else:
        'prediction'
        outputs=UpSampling2D(size=(_up_rate, _up_rate), name="lcz")(outputs)
        return outputs, conv3, conv2


def sen2mt_net(input_size = (128,128,10), bn=1, residual=0, flow=2, taskAttation=0, shareNet=0, fusion=1, lczFeatureReturn=0):

    inputs = Input(input_size)

    if flow ==2:
        o_0, o= sen2_LCZ_HSE(inputs, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)
        model = Model(inputs, [o_0, o], name='mtsNN')

    if flow ==1:
        o1 = sen2_LCZ_HSE(inputs, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)
        model = Model(inputs, o1, name='lczNN')

    if flow ==0:
        o0 = sen2_LCZ_HSE(inputs, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)
        model = Model(inputs, o0, name='hseNN')

    return model


#################################################################learning weight nn = 4
#https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
def sen2mt_net_mt_lw(input_size = (128,128,10), y_0_Shape=(64, 64, 1), y_1_Shape=(128, 128, 17), bn=1, residual=0, flow=2, taskAttation=0, shareNet=0, fusion=1, lczFeatureReturn=0):
    inp = Input(shape=input_size, name='inp')

    y1_pred, y2_pred = sen2_LCZ_HSE(inp, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)

    y1_true = Input(shape=y_0_Shape, name='y1_true')
    y2_true = Input(shape=y_1_Shape, name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])

    print(out.shape)

    return Model([inp, y1_true, y2_true], out)

#without lw
def sen2mt_net_mt_p2f(input_size = (128,128,10), y_0_Shape=(64, 64, 1), y_1_Shape=(128, 128, 17), bn=1, flow=2, taskAttation=0, shareNet=0, fusion=1, lczFeatureReturn=1):
    inp = Input(shape=input_size, name='inp')

    y1_pred, outputs, outputs_32, outputs_16, outputs_8, conv3, conv2, conv1 = sen2_LCZ_HSE(inp, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)

    y1_true = Input(shape=y_0_Shape, name='y1_true')
    y2_true = Input(shape=y_1_Shape, name='y2_true')

    concat = multiply([y1_true, conv1])
    o_1 = Conv2D(17, (1, 1), activation='softmax', kernel_initializer='he_normal')(concat)
    _up_rate=int(int(input_size[1])/int(o_1.shape[1]))
    "because of size of gt"
    o_1=UpSampling2D(size=(_up_rate, _up_rate))(o_1)

    y2_pred=Average(name="lcz_final")([o_1, outputs, outputs_32, outputs_16, outputs_8])

    return Model([inp, y1_true, y2_true], [y1_pred, y2_pred])


def sen2mt_net_mt_lw_p2f_1(input_size = (128,128,10), y_0_Shape=(64, 64, 1), y_1_Shape=(128, 128, 17), bn=1, residual=0, flow=2, taskAttation=0, shareNet=0, fusion=1, lczFeatureReturn=1):
    inp = Input(shape=input_size, name='inp')

    y1_pred, outputs, outputs_32, outputs_16, outputs_8, conv3, conv2, conv1 = sen2_LCZ_HSE(inp, flow=flow, taskAttation=taskAttation, bn=bn, shareNet=shareNet, fusion=fusion, lczFeatureReturn=lczFeatureReturn)

    y1_true = Input(shape=y_0_Shape, name='y1_true')
    y2_true = Input(shape=y_1_Shape, name='y2_true')
    #

    concat = multiply([y1_true, conv1])
    o_1 = Conv2D(17, (1, 1), activation='softmax', kernel_initializer='he_normal')(concat)
    _up_rate=int(int(input_size[1])/int(o_1.shape[1]))
    "because of size of gt"
    o_1=UpSampling2D(size=(_up_rate, _up_rate))(o_1)


    y2_pred=Average(name="lcz_")([o_1, outputs, outputs_32, outputs_16, outputs_8])#


    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])

    return Model([inp, y1_true, y2_true], out)


def modelPredict_lw(model, x_test, layerName='lcz'):

    inputs1 = np.zeros((x_test.shape[0],64,64,1))
    inputs2 = np.zeros((x_test.shape[0],128,128,17))

    intermediate_hse_model = Model(inputs=model.input,outputs=model.get_layer('hse').output)
    y0 = intermediate_hse_model.predict([x_test, inputs1, inputs2], batch_size = 16, verbose=1)

    intermediate_lcz_model = Model(inputs=model.input,outputs=model.get_layer(layerName).output)
    y1 = intermediate_lcz_model.predict([x_test, inputs1, inputs2], batch_size = 16, verbose=1)

    return y0, y1
####################################################
'''
' Huber loss.
' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
' https://en.wikipedia.org/wiki/Huber_loss
'''
def huber_loss(y_true, y_pred, clip_delta=0.7):
	error = y_true - y_pred
	cond  = tf.keras.backend.abs(error) < clip_delta

	squared_loss = 0.5 * tf.keras.backend.square(error)
	linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

	diff = tf.where(cond, squared_loss, linear_loss)

	return K.mean(diff, axis=-1)

def huber_loss_weight(y_true, y_pred, clip_delta=0.7):
	error = y_true - y_pred
	cond  = tf.keras.backend.abs(error) < clip_delta

	squared_loss = 0.5 * tf.keras.backend.square(error)
	linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

	diff = tf.where(cond, squared_loss, linear_loss)  * (K.exp(y_true))

	return K.mean(diff, axis=-1)


def mean_absolute_error_weight(y_true, y_pred):
  diff = K.abs(y_true - y_pred) * (K.exp(y_true))
  return K.mean(diff, axis=-1)


def mean_square_error_weight(y_true, y_pred):
  diff = K.abs(y_true - y_pred)* K.abs(y_true - y_pred) * (K.exp(y_true))
  return K.mean(diff, axis=-1)
