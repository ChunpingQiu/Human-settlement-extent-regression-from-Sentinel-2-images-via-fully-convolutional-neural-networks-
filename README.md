# multi-task for HSE regression and LCZ classification

<!---
## main contents
- data preparation
- HSE model training
- HSE model evaluation
- HSE regression in unseen area
- mtl
-->

## datasets
- HSE reference: https://drive.google.com/drive/folders/1n2LGeGAv_O2cvxAJnSGNRUI4FMsm4psa?usp=sharing
- (small) sample data: mtl_SampleData, https://drive.google.com/drive/folders/15Q63JR4X4wT9TBUZQxsc5eEjVlYYygeX?usp=sharing

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training mtl and single task
  ├── spec-file_conda_env.txt - information about the conda environment
  │
  ├── model/ - definition about the models
  │   ├── plotModel.py - plot the models
  │   ├── modelS.py - select the models according to the setup(single task or mtl)
  │   ├── model_sep_cbam.py - definition of the mtl framework
  │   └── ...
  │   
  │    
  ├── dataPrepare/ - small utility functions
  │   ├── dataGener.py - data generator for model.fit_generator
  │   ├── lr.py - step_decay_schedule of learning rate
  │   ├── check_created_Patches.ipynb - check and visulize the patches
  │   └── ...
  ├── HSE_regressionTask
  │   ├── modelS_hse.py -models (and setups) to be used and tested
  │   ├── plotModel.py - visualize the models
  │   ├── test_hse.py - test different models
  │   └── train_hse.py - train different models
  │   
  │
  └── modelPredict/ - predict with the trained model
      ├── .py - get hse predictions with the test dataset        
      ├── img2map.py - predict hse and lcz using the trained models for a image file (much larger than the patches)
      └── ...       

  ```
## Usage

### Investigations on HSE regression task
- define/setup models in modelS_hse.py: `CUDA_VISIBLE_DEVICES=0 python plotModel.py`
- train models of different setups: `CUDA_VISIBLE_DEVICES=N python train_hse.py --methods4test sen2mt_net_Loss_mae sen2mt_net_Loss_mse --folderData './mtl_SampleData/patches/' --saveFolder './results/'`
- test the trained models: `CUDA_VISIBLE_DEVICES=N python test_hse.py --methods4test sen2mt_net_Loss_mae sen2mt_net_Loss_mse --folderData './mtl_SampleData/patches/test/' --modelPath './results/'`
[//]: # (- predictions from tif files:)
[//]: # (### Investigations on LCZ classification task)

### Investigations on MTL
- define/setup models in modelS.py `CUDA_VISIBLE_DEVICES=0 python plotModel.py`
- `CUDA_VISIBLE_DEVICES=N python train.py --methods4test w_learned --folderData './mtl_SampleData/patches/' --saveFolder './results/'`
- predictions from tif files: `CUDA_VISIBLE_DEVICES=0 python img2map.py --methods4test w_learned --modelPath './results/' --tifFile './mtl_SampleData_tif/henan_2017_sentinel_22.tif' --modelWeights "weights.best_lcz"`

## td list
- [ ] visualize predictions with gee (currently the predicted .tif files can be visualized in Qgis)
- [ ] build and test on a small but well distributed dataset for further methodology development
<!---
[//]: # (- [x] predict with the trained model)
- [x] test different models with the same data
- [x] training different models under the same configuration
- [x] check created patches
- [x] from images to patches
-->
