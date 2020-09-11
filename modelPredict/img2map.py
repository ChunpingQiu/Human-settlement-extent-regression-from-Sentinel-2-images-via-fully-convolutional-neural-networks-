# @Date:   2018-08-01T11:06:48+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-04-19T21:11:07+02:00

import sys
sys.path.insert(0, '../model')
import modelS
from img2mapC import img2mapC
import numpy as np
# import time

from keras.models import load_model
import h5py
import argparse

from keras import backend as K
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
###################################################################

"set up of Parameters"
##############
parser = argparse.ArgumentParser()
parser.add_argument('--methods4test', nargs='+', type=str)
parser.add_argument('--tifFile', default='./mtl_SampleData_tif/henan_2017_sentinel_22.tif')
parser.add_argument('--modelPath', default='./results/')
parser.add_argument('--modelWeights', default="weights.best_lcz")#weights.best_lcz
args = parser.parse_args()
###################################################################
MapfileD=args.modelPath+args.modelWeights+'/'
lr=0.002
patch_shape = (128, 128, 10)
batchS=8

params = {'dim_x': patch_shape[0],
		   'dim_y': patch_shape[1],
		   'step': patch_shape[0],
		   'Bands': [1,2,3,4,5,6,7,8,11,12],
		   'scale':10000.0}

img2mapCLass=img2mapC(**params);

if not os.path.exists(MapfileD):
	os.makedirs(MapfileD)

for nn in args.methods4test:

	print('!!!!!!!!!!!!!!!!!!!!!!!!!!nn:', nn)

	#for idCity in [0]:

	model, outNumber, _ = modelS.modelSelection(nn, patch_shape, lr)

	modelName =  args.modelPath +str(lr)+"_"+str(nn)+"_"+str(batchS)+ args.modelWeights +".hdf5"
	model.load_weights(modelName, by_name=False)
	#print(modelName)
	#print(params['Bands'])
	#print(files)
	mapFile = MapfileD+ os.path.basename(args.tifFile)[:-4]+'_'+ str(nn)
	img2mapCLass.img2Bdetection_ovlp(args.tifFile, model, mapFile, out=outNumber, nn=nn)
