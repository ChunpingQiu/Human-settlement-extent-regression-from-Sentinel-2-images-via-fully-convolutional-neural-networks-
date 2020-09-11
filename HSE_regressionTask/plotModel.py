# @Date:   2018-09-08T14:08:13+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-04-19T13:49:08+02:00

import sys
from pathlib import Path
from keras.utils import plot_model
import modelS_hse
import keras.backend as K

'''
plot the models for visualzation and check
'''

import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)

file='./modelFigure/'

# 0. parameters
patch_shape = (128, 128, 10)
learnRate = 0.0002#the same as isprs

for nn in ["sen2mt_net", "unet", "dlab"]:#

    print('###INFO:            nn #:',nn)

    model, _ = modelS_hse.modelSelection(nn, patch_shape, learnRate)

    model.summary()
    plot_model(model, to_file=file+"Model="+str(nn)+'-Task=hse.png', show_shapes='True')
