# @Date:   2019-05-13
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-04-18T17:15:28+02:00


###
import sys
sys.path.insert(0, '../dataPrepare')
from dataGener import DataGenerator
import lr

import modelS_hse

import numpy as np
import glob, glob2
import time
import scipy.io as sio

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

import argparse

import tensorflow as tf
from keras import backend as K
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2#0.41
session = tf.Session(config=config)

#################################################################
"set up of Parameters"
##############
parser = argparse.ArgumentParser()
parser.add_argument('--methods4test', nargs='+', type=str)
parser.add_argument('--folderData', default='./mtl_SampleData/')
parser.add_argument('--saveFolder', default='./results/')
args = parser.parse_args()
# ##################################################################
lr_sched = lr.step_decay_schedule(initial_lr=0.002, decay_factor=0.75, step_size=2)
patch_shape = (128, 128, 10)
epochs = 100
batch_size = 8
learnRate = 0.002
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
###################################################################
if not os.path.exists(args.saveFolder):
	os.makedirs(args.saveFolder)

'number of training and validate samples'
valNum=0
traNum=0
for sea in ['autumn', 'summer', 'spring']:
	mat = sio.loadmat(args.folderData+'vali/'+'_'+sea+'patchNum.mat')
	patchNum=mat['patchNum']*1
	print(patchNum)
	valNum=valNum+np.sum(patchNum[0,:]);

	mat = sio.loadmat(args.folderData+'trai/'+'_'+sea+'patchNum.mat')
	patchNum=mat['patchNum']*1
	print(patchNum)
	traNum=traNum+np.sum(patchNum[0,:]);

#validata samples
fileVal = glob2.glob(args.folderData+'vali/' +'*.h5')
#training samples
fileTra = glob2.glob(args.folderData+'trai/' +'*.h5')
# shuffle(fileVal)
# shuffle(fileTra)
print('###INFO:            train files:', len(fileTra))
print('###INFO:            vali files:',  len(fileVal))
print('###INFO:            train patch:', traNum)
print('###INFO:            vali patch:',  valNum)

###################################################################
for nn in args.methods4test:

	print("###INFO:            !!!!!!!!!!!!!!!!!!!!!!!!!!", nn)

	timeCreated=time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))

	fileSave=  args.saveFolder + str(learnRate) + '_' + nn + '_' + str(batch_size)
	checkpoint = ModelCheckpoint(fileSave+"weights.best.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

	tbCallBack = TensorBoard(log_dir=args.saveFolder+'logs' + '_' + nn + '_' + str(batch_size) + '_' + timeCreated,  # log 目录
					 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
	#                  batch_size=32,     # 用多大量的数据计算直方图
					 write_graph=True,  # 是否存储网络结构图
					 write_grads=True, # 是否可视化梯度直方图
					 write_images=True,# 是否可视化参数
					 embeddings_freq=0,
					 embeddings_layer_names=None,
					 embeddings_metadata=None)
	###################################################################
	model, dataGenerFlow = modelS_hse.modelSelection(nn, patch_shape, learnRate)

	# Generators
	params = {'dim_x': patch_shape[0],
			  'dim_y': patch_shape[1],
			  'dim_z': patch_shape[2],
			  'batch_size':batch_size,
			  'flow': dataGenerFlow}
	tra_generator = DataGenerator(**params).generate(fileTra)
	val_generator = DataGenerator(**params).generate(fileVal)

	start = time.time()
	model.fit_generator(generator = tra_generator,
					steps_per_epoch = traNum//batch_size, epochs = epochs,
					validation_data = val_generator,
					validation_steps = valNum//batch_size,
					callbacks = [checkpoint, tbCallBack, early_stopping, lr_sched], max_queue_size = 50, verbose=1)
	end =time.time()

	trainingTime=end-start;
	savedModel = fileSave + 'model.final_' +'.h5'
	model.save_weights(savedModel)
	sio.savemat((fileSave+'_trainingTime_.mat'), {'trainingTime':trainingTime})
