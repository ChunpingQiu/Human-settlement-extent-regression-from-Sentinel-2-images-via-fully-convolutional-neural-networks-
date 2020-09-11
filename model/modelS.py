# @Date:   2020-02-22T17:50:48+01:00
# @Last modified time: 2020-09-06T13:53:57+02:00

import sys
sys.path.insert(0, './model')
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Nadam

import model_sep_cbam

def modelSelection(nn, patch_shape,learnRate):

	nadam = Nadam(lr=learnRate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

	outNumber = 2 ;
	###################################################"w_learned_p2f" "w_learned" "w_11"

	"lcz task"
	if nn=="bs_lcz":
		import model_sep_cbam
		outNumber = 1
		model = model_sep_cbam.sen2mt_net(input_size=patch_shape, flow = 1, taskAttation=1)
		model.compile(optimizer=nadam, loss="categorical_crossentropy", metrics=['accuracy'])
		dataGenerFlow=1

	"mtl, with weight 1 1"
	if nn == "w_11":
		import model_sep_cbam
		lossWeights = {"hse": 1.0, "lcz": 1.0}
		model = model_sep_cbam.sen2mt_net(input_size=patch_shape, flow = 2, taskAttation=1)
		losses = {
			"hse": model_sep_cbam.mean_absolute_error_weight,
			"lcz": "categorical_crossentropy",
		}
		model.compile(optimizer=nadam, loss= losses, loss_weights=lossWeights, metrics=['mae','accuracy'])
		dataGenerFlow=2

	"mtl, learned weight"
	if nn=="w_learned":
		import model_sep_cbam
		model = model_sep_cbam.sen2mt_net_mt_lw(input_size=patch_shape, flow = 2, taskAttation=1)
		model.compile(optimizer=nadam, loss=None, metrics=['mae','accuracy'])
		model.metrics_names.append("hse_loss")
		model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[0])
		model.metrics_names.append("lcz_loss")
		model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[1])
		model.metrics_names.append("v1")
		model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[2])
		model.metrics_names.append("v2")
		model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[3])
		dataGenerFlow=3

	# "mtl, learned weight; p2f"
	# if nn=="w_learned_p2f":
	# 	import model_sep_cbam
	# 	model = model_sep_cbam.sen2mt_net_mt_lw_p2f_1(input_size=patch_shape, flow = 2, taskAttation=1)
	# 	model.compile(optimizer=nadam, loss=None, metrics=['mae','accuracy'])#metrics
	# 	model.metrics_names.append("hse_loss")
	# 	model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[0])
	# 	model.metrics_names.append("lcz_loss")
	# 	model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[1])
	# 	model.metrics_names.append("v1")
	# 	model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[2])
	# 	model.metrics_names.append("v2")
	# 	model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[3])
	# 	dataGenerFlow=3
	#
	# # if nn==37:
	# # 	import model_sep_cbam
	# # 	lossWeights = {"hse": 1.0, "lcz_final": 1.0}
	# # 	model = model_sep_cbam.sen2mt_net_mt_p2f(input_size=patch_shape, flow = 2, taskAttation=1)
	# # 	losses = {
	# # 		"hse": model_sep_cbam.mean_absolute_error_weight,
	# # 		"lcz_final": "categorical_crossentropy",
	# # 	}
	# # 	model.compile(optimizer=nadam, loss= losses, loss_weights=lossWeights, metrics=['mae','accuracy'])
	# # 	dataGenerFlow=3
	#
	# "mtl, learned weight; p2f; no attention"
	# if nn=="w_learned_p2f_noCBAM":
	# 	import model_sep_cbam
	# 	model = model_sep_cbam.sen2mt_net_mt_lw_p2f_1(input_size=patch_shape, flow = 2, taskAttation=0)
	# 	# model.compile(optimizer=nadam, loss=None, metrics=['mae','accuracy'])#metrics
	# 	# model.metrics_names.append("hse_loss")
	# 	# model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[0])
	# 	# model.metrics_names.append("lcz_loss")
	# 	# model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[1])
	# 	# model.metrics_names.append("v1")
	# 	# model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[2])
	# 	# model.metrics_names.append("v2")
	# 	# model.metrics_tensors.append(model.get_layer('custom_multi_loss_layer_1').output[3])
	# 	dataGenerFlow=3

	return model, outNumber, dataGenerFlow
