# @Date:   2018-07-16T20:51:18+02:00
# @Email:  chunping.qiu@tum.de
# @Last modified time: 2020-04-19T21:10:48+02:00


import sys
import os
import numpy as np
from skimage.util.shape import view_as_windows
import skimage.measure

import glob
from osgeo import gdal,osr
import glob2
from scipy import stats
import scipy.ndimage

from keras import backend as K
from keras.models import Model
from keras.layers import Input

sys.path.insert(0, '../model')
from model_sep_cbam import modelPredict_lw

'load img tif and get patches from it;  save as tif file after getting predictions'
class img2mapC(object):

  def __init__(self, dim_x, dim_y, step, Bands, scale):
	  self.dim_x = dim_x#shape of the patch
	  self.dim_y = dim_y
	  self.step = step#step when create patches from images (in pixel): step
	  self.Bands = Bands#bands selected from the image files, list
	  self.scale = scale#the number used to divided the pixel value by


  '''
    # from a multi bands mat to patches, without considering the nan area
    # input:
            imgMat: image of multiple bands
    # output:
            patch: the patch of this input image, to feed to the classifier
            R: the row number of the patches
            C: the colum number of the patches
  '''
  def Bands2patches_all(self, imgMat, upSampleR=1):

	  for band in np.arange(imgMat.shape[2]):
		  arr = imgMat[:,:,band]
		  if upSampleR!=1:
		            arr=scipy.ndimage.zoom(arr, [upSampleR,  upSampleR], order=1)#Bilinear interpolation would be order=1
		  patch0, R, C= self.__img2patch(arr)#'from band to patches'

		  if band==0:
			  patch=np.zeros(((patch0.shape[0]), self.dim_x, self.dim_y, imgMat.shape[2]), dtype=imgMat.dtype);

		  #print('self.scale', self.scale)
		  if self.scale == -1:#scale with a fucntion
			  patch[:,:,:,band]=self.scaleBand(patch0)
		  else:
			  patch[:,:,:,band]=patch0/self.scale ;

	  return patch, R, C


  '''
    # load all relevent bands of a image file
    # input:
            imgFile: image file
    # output:
            prj: projection data
            trans: projection data
            matImg: matrix containing the bands of the image
  '''
  def loadImgMat(self, imgFile):
	  src_ds = gdal.Open( imgFile )

	  if src_ds is None:
		  print('Unable to open INPUT.tif')
		  sys.exit(1)
	  prj=src_ds.GetProjection()
	  trans=src_ds.GetGeoTransform()
	  #print("[ RASTER BAND COUNT ]: ", src_ds.RasterCount)
	  #print(prj)
	  #print(trans)

	  #print(self.Bands)

	  bandInd=0
	  print(self.Bands)
	  for band in self.Bands:
		  band += 1
		  srcband = src_ds.GetRasterBand(band)

		  if srcband is None:
			  print('srcband is None'+str(band)+imgFile)
			  continue

		  #print('srcband read:'+str(band))

		  arr = srcband.ReadAsArray()
		  # print(arr.dtype, np.unique(arr))

		  if bandInd==0:
			  R=arr.shape[0]
			  C=arr.shape[1]
			  #print(arr.shape)
			  matImg=np.zeros((R, C, len(self.Bands)), dtype=np.float32);
		  matImg[:,:,bandInd]=np.float32(arr)


		  bandInd += 1

	  return prj, trans, matImg


  '''
    # use the imgmatrix to get patches
    # input:
            mat: a band of the image
    # output:
            patches: the patch of this input image, to feed to the classifier
            R: the number of the small patches within in the whole mat in row direction
            C: the number of the small patches within in the whole mat in col direction
  '''
  def  __img2patch(self, mat):

      #mat=np.pad(mat, ((np.int(self.dim_x_img/2), np.int(self.dim_x_img/2)), (np.int(self.dim_y_img/2), np.int(self.dim_y_img/2))), 'reflect')

      window_shape = (self.dim_x, self.dim_x)#self.dim_x_img

	  #window_shape = (self.dim_x, self.dim_y)#self.dim_x_img
      B = view_as_windows(mat, window_shape, self.step)#B = view_as_windows(A, window_shape,2)
      #print(B.shape)

      patches=np.reshape(B, (-1, window_shape[0], window_shape[1]))
      #print(patches.shape)

      #patches=scipy.ndimage.zoom(patches, [1, (self.dim_x/window_shape[0]),  (self.dim_y/window_shape[1])], order=1)#Bilinear interpolation would be order=1

      R=B.shape[0]#the size of the final map
      C=B.shape[1]

      return patches, R, C


  '''
    # save a map as tif
    # input:
            mat: the matrix to be saved
            prj: projection data
            trans: projection data
            mapFile: the file to save the produced map
    # output:
            no
  '''
  def predic2tif(self, mat, prj, geotransform, mapFile):

	  R=mat.shape[0]
	  C=mat.shape[1]

	  # totalNum=R*C;
      # trans = (trans0, trans1, 0, trans3, 0, trans5)
	  # xres =trans[1]*1
	  # yres= trans[5]*1
	  # geotransform = (trans[0]+xres*(1-1)/2.0, xres, 0, trans[3]+xres*(1-1)/2.0, 0, yres)

	  dimZ=mat.shape[2]

	 # create the dimZ raster file
	  dst_ds = gdal.GetDriverByName('GTiff').Create(mapFile, C, R, dimZ, gdal.GDT_UInt16)#gdal.GDT_Byte .GDT_Float32 gdal.GDT_Float32
	  dst_ds.SetGeoTransform(geotransform) # specify coords
	  dst_ds.SetProjection(prj)

	  for i in np.arange(dimZ):
		  map=mat[:,:,i]
		  dst_ds.GetRasterBand(int(i+1)).WriteArray(map)   # write band to the raster
	  dst_ds.FlushCache()                     # write to disk
	  dst_ds = None


  def scaleBand(self,patches):
      patches_=np.zeros(patches.shape, dtype=np.float32)
      #for b in np.arange(patches.shape[-1]):

      patch=patches.reshape(-1,1)
      #print(patch.shape)
      scaler = StandardScaler().fit(patch)
      #print(scaler.mean_.shape)
      patches_=scaler.transform(patch).reshape(patches.shape[0],patches.shape[1], patches.shape[2])

      return patches_

  '''
    # predict from image file and save as tif files
    # input:
            file: *.tif
            model: mdoel with trained weights
            mapFile: filename for the output
            out: 2 for multi task learning and 1 for single task
            nn: mdoel name, lcz task ends with "_lcz"
    # output:
            none
  '''
  def img2Bdetection_ovlp(self, file, model, mapFile, out=1, nn=0):
      prj, trans, img= self.loadImgMat(file)
      R=img.shape[0]
      C=img.shape[1]
      # print('img:', R, C)

      paddList=[0,32,64,96]

      for padding in paddList:#this means that the final prediciton is mean of four predictions with different overlaps

          if padding==0:
              img1=img
          else:
              img1=np.pad(img, ((padding, 0), (padding, 0), (0,0)), 'reflect')
          print(img1.shape)
          x_test, mapR, mapC = self.Bands2patches_all(img1)
          print('x_test:', x_test.shape)

          "multi task"
          if out==2:

              if nn == "w_learned":
                  print('get the intermidate output as the prediciton:')
                  y0,y1=modelPredict_lw(model, x_test, layerName='lcz')
                  print(y0.shape, y1.shape)

              elif nn == "w_learned_p2f":
                  print('prediction to feature for prediciton:')
                  y0,y1=modelPredict_lw(model, x_test, layerName='lcz_')

              elif nn == "w_11":
                  print('direct out 2 prediction:')
                  y0, y1 = model.predict(x_test, batch_size = 16, verbose=1)

              else:
                  print('!!!!!!!!!!!!!!!!!!!!!!!!!wrong nn:')

              C0 =self.pro_from_x(mapR, mapC, y0)
              mapPatch_shape_0=y0.shape[1]

              C1 =self.pro_from_x(mapR, mapC, y1)
              mapPatch_shape_1=y1.shape[1]

              OS0 = np.int( self.dim_x/ mapPatch_shape_0 )   #ratio between the input and the output
              OS1 = np.int( self.dim_x/ mapPatch_shape_1 )

              if padding==0:
                  r0=C0.shape[0]
                  c0=C0.shape[1]
                  Pro0=C0[0:(r0-mapPatch_shape_0),0:(c0-mapPatch_shape_0),:]
                  # Pro0=C0

                  r1=C1.shape[0]
                  c1=C1.shape[1]
                  Pro1=C1[0:(r1-mapPatch_shape_1),0:(c1-mapPatch_shape_1),:]
                  # Pro1=C1
              else:
                  Pro0=Pro0+C0[np.int(padding/OS0):(r0-mapPatch_shape_0+np.int(padding/OS0)), np.int(padding/OS0):(c0-mapPatch_shape_0+np.int(padding/OS0)), :]
                  Pro1=Pro1+C1[np.int(padding/OS1):(r1-mapPatch_shape_1+np.int(padding/OS1)), np.int(padding/OS1):(c1-mapPatch_shape_1+np.int(padding/OS1)), :]
                  # Pro0=Pro0+C0[np.int(padding/OS0):(r0+np.int(padding/OS0)), np.int(padding/OS0):(c0+np.int(padding/OS0)), :]
                  # Pro1=Pro1+C1[np.int(padding/OS1):(r1+np.int(padding/OS1)), np.int(padding/OS1):(c1+np.int(padding/OS1)), :]

          "single task"
          if out==1:
              y = model.predict(x_test, batch_size = 16, verbose=1)

              mapPatch_shape=y.shape[1]
              C =self.pro_from_x(mapR, mapC, y)
              OS = np.int( self.dim_x/ mapPatch_shape )   #ratio between the input and the output
              if padding==0:
                  r=C.shape[0]
                  c=C.shape[1]
                  Pro=C[0:(r-mapPatch_shape),0:(c-mapPatch_shape),:]#shape of the final prediction
                  # Pro=C#shape of the final prediction
              else:
                  Pro=Pro+C[np.int(padding/OS):(r-mapPatch_shape+np.int(padding/OS)), np.int(padding/OS):(c-mapPatch_shape+np.int(padding/OS)), :]
                  # Pro=Pro+C[np.int(padding/OS):(r+np.int(padding/OS)), np.int(padding/OS):(c+np.int(padding/OS)), :]

      if out==1:
          if nn[-3:]=="lcz":
              ratio=10
              Pro=skimage.measure.block_reduce(Pro, (ratio, ratio,1), np.mean)
              self.save_pre_pro(prj, trans, Pro, mapFile+'_lcz', ratio)
          else:
              ratio=self.dim_x / mapPatch_shape
              self.save_pre_pro(prj, trans, Pro, mapFile+'_hse', ratio)

      if out==2:
          ratio=self.dim_x / mapPatch_shape_0
          self.save_pre_pro(prj, trans, Pro0, mapFile+'_hse', ratio)

          ratio=10
          Pro1=skimage.measure.block_reduce(Pro1, (ratio, ratio,1), np.mean)
          self.save_pre_pro(prj, trans, Pro1, mapFile+'_lcz', ratio)


  '''
    # get y of the targeting shape
    # input:
            y: prediction from the model, n*h*w*C
            mapR: the number of the small patches within in the whole mat in row direction
            mapC: the number of the small patches within in the whole mat in col direction
    # output:
            C: prediction arranged as the whole mat (before view_as_windows)
  '''
  def pro_from_x(self, mapR, mapC, y):

      # mapPatch_shape=y.shape[1]
      print('class num:', y.shape[-1])

      B_=np.reshape(y, (mapR, mapC, y.shape[1], y.shape[2], y.shape[-1]))
      print('B_.shape', B_.shape)
      del y

      C=np.zeros((B_.shape[0]*B_.shape[2], B_.shape[1]*B_.shape[3], B_.shape[4]), dtype=float)
      for dim in np.arange(B_.shape[4]):
          B_1=B_[:,:,:,:,dim]
          C[:,:,dim]=B_1.transpose(0,2,1,3).reshape(-1, B_1.shape[1]*B_1.shape[3])
          del B_1
      return C#, mapPatch_shape

  ''' save predictions and pro'''
  def save_pre_pro(self, prj, trans, Pro, mapFile, ratio):

      mapPro=np.zeros((Pro.shape[0], Pro.shape[1], 1), dtype=np.uint16)

      'regression or not'
      if Pro.shape[-1]==1:
          mapPro= Pro/4.0*100;
          print(np.mean(mapPro), np.amax(mapPro))
      else:
          y=Pro.argmax(axis=2)+1
          mapPro[:,:,0]=y;

      "setup the geo Ref of the output"
      print('downsampling by: ', ratio)
      trans0 =trans[0]+trans[1]*(ratio-1)/2.0
      trans3= trans[3]+trans[5]*(ratio-1)/2.0
      trans1 =trans[1]* ratio
      trans5= trans[5]* ratio
      trans = (trans0, trans1, 0, trans3, 0, trans5)

      print(prj, trans)

      self.predic2tif(mapPro, prj, trans, mapFile+'.tif')


  ''' generate class prediction from the input samples'''
  def predict_classes(self, x):
	  y=x.argmax(axis=1)+1
	  return y
