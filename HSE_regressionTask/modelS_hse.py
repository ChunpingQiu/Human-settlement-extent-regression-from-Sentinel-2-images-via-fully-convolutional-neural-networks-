# @Date:   2020-02-22T17:50:48+01:00
# @Last modified time: 2020-04-19T13:47:57+02:00

import sys

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Nadam

sys.path.insert(0, './models')

sys.path.insert(0, '../model')
import model_sep_cbam
######################################################################
def modelSelection(model_name, patch_shape, learnRate):

	nadam = Nadam(lr=learnRate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

	dataGenerFlow=0

	# print(model_name is "sen2mt_net-Loss_mae")
 	##################################################

	if model_name == "sen2mt_net_Loss_mae":
		model = model_sep_cbam.sen2mt_net(input_size=patch_shape, flow = dataGenerFlow, taskAttation=1)
		model.compile(optimizer=nadam, loss="mean_absolute_error", metrics=['mae'])

	if model_name == "sen2mt_net_Loss_mse":
		model = model_sep_cbam.sen2mt_net(input_size=patch_shape, flow = dataGenerFlow, taskAttation=1)
		model.compile(optimizer=nadam, loss="mean_squared_error", metrics=['mae'])

	if model_name == "sen2mt_net":
		model = model_sep_cbam.sen2mt_net(input_size=patch_shape, flow = dataGenerFlow, taskAttation=0)
		model.compile(optimizer=nadam, loss=model_sep_cbam.mean_absolute_error_weight, metrics=['mae'])

	if model_name == "dlab":
		import deepLabV3_adapted
		model = deepLabV3_adapted.dl_net(input_size=patch_shape)
		model.compile(optimizer=nadam, loss=model_sep_cbam.mean_absolute_error_weight, metrics=['mae'])

	if model_name == "unet":
		import unet
		model = unet.unet(input_size=patch_shape)
		model.compile(optimizer=nadam, loss=model_sep_cbam.mean_absolute_error_weight, metrics=['mae'])

	return model, dataGenerFlow
