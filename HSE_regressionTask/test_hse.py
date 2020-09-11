# @Date:   2020-04-06T13:22:37+02:00
# @Last modified time: 2020-04-20T18:41:29+02:00


import sys

import modelS_hse

import h5py
import numpy as np
import glob
import scipy.io as sio
# import skimage.measure
from sklearn.metrics import mean_absolute_error

import argparse

import tensorflow as tf
from keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4#0.41
session = tf.Session(config=config)

#################################################################
"set up of Parameters"
##############
parser = argparse.ArgumentParser()
"methods to be tested"
parser.add_argument('--methods4test', nargs='+', type=str)
'testing samples'
parser.add_argument('--folderData', default='./mtl_SampleData_testHSE/')
'path to the trained models'
parser.add_argument('--modelPath', default='./results/')
args = parser.parse_args()
# print(args.methods4test)
#################################
lr=0.002
patch_shape = (128, 128, 10)
batch_size = 8
###################################################################
file0=args.modelPath+'test_EU_results/'
if not os.path.exists(file0):
	os.makedirs(file0)

fileVal = glob.glob(args.folderData +'*.h5')
print('###INFO:            files #:', len(fileVal))

for file in fileVal:

	'read the file'
	print(file, os.path.basename(file)[:-3])
	hf = h5py.File(file, 'r')
	x_tst=np.array(hf.get('x'))

	y_tst_0=np.array(hf.get('y'))
	# y_tst_0=np.int8(y_tst_0.argmax(axis=-1))
	# print(y_tst_0.shape, y_tst_0.dtype)

	hf.close()

	for nn in args.methods4test:

		model, dataGenerFlow = modelS_hse.modelSelection(nn, patch_shape, lr)

		fileSave=  file0  + str(nn) + '_' + str(batch_size) + os.path.basename(file)[:-3]

		modelName =  args.modelPath +str(lr) +"_"+str(nn)+"_"+str(batch_size)+"weights.best.hdf5"
		print('###INFO:            modelName #:', modelName)
		model.load_weights(modelName, by_name=False)

		y_pre=model.predict(x_tst)

		y_pre=y_pre*100

		print('###INFO:            shape before save:', y_tst_0.shape, y_tst_0.dtype, y_pre.shape, y_pre.dtype)
		sio.savemat((fileSave+str(lr)+'_pre_hse.mat'), {'y_pre':y_pre, 'y_tst':y_tst_0})

		print("###INFO:            mean_absolute_error of this file:")
		print(mean_absolute_error(y_pre.flatten(), y_tst_0.flatten()))
