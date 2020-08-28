# Work Log

# Goal
Accuracy > 71 %

## TODO:
1. Weighted loss (V)
2. Change the class label (V)
3. Confusion matrix (V)
4. Check age distribution in data set (V)
5. Check [kaggle notebook](https://www.kaggle.com/c/diabetic-retinopathy-detection/notebooks)
   1. Attention Model (V)
   2. use large resolution
6. Try increase the image resolution with smaller batch size, grad accumulated method
   1. suggest 640,original 800 (V)
7. Try EficientNet (-)
8. Check train-loss and val-loss (V)
9. Dataset question: Patients are all sick? (-)


## 8/28

### Exp32
* epoch 15
* batch_size 16 
* image_size 300
* model se_resnext101_32x4d + attension
* optim SGD
* output class: 5
* loaded model: `31_se_resnext101_32x4d_best.pth`
* unfreeze
Results
* 0:59:12.7
* 77.81 % accuracy 
* save as: `32_se_resnext101_32x4d_best.pth`
Comment
TODO:
Seems can keep training?


### Exp31
* epoch 10
* batch_size 16 
* image_size 300
* model se_resnext101_32x4d + attension
* optim SGD
* output class: 5
* loaded model: 18_se_resnext101_32x4d_best.pth
* freeze se_resnext101 
Results
* 0:39:23.3
* 75.95 % accuracy 
* save as: `31_se_resnext101_32x4d_best.pth`
Comment
Using the best head extractor, to get the freeze result.


## 8/27
### Exp26
* epoch 10
* batch_size 16 
* image_size 300
* model se_resnext101_32x4d + attension
* optim Adam
* output class: 5
* attension mode inherence freeze se_resnext101_32x4d (imagenet pretrained)
Results
* 0:39:37.5
* 76.23 % accuracy
* save as: `26_se_resnext101_32x4d_best.pth`
Comment
training faster than original fc final two layers, and achieve higher accuracy, and lower image resolution, training loss converge around 300 iterations, achieve highest accuracy at 7th epoch. (Compared with `14_se_resnext101_32x4d_best.pth`)


## 8/26
### Exp18
* epoch 15
* batch_size 8 (grad acc) 
* image_size 800
* model se_resnext101_32x4d
* optim SGD
* output class: 5
* loaded model: 16_se_resnext101_32x4d_best.pth (75%)
* unfreeze
Results
* 11:40:56.4
* 78.56 % accuracy
* save as: `18_se_resnext101_32x4d_best.pth`


## 8/24
### Reclass (20 yr seperation)
### Weighted loss
### Exp14
* epoch 10
* batch_size 16 (grad acc)
* image_size 800
* model se_resnext101_32x4d
* optim Adam
* output class: 5
* freeze
Results
* 2:31:37.3
* 71.24 % accuracy
* save as: `14_se_resnext101_32x4d_best.pth`

### Exp13
* epoch 15
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* optim SGD
* output class: 5
* loaded model: 12_se_resnext101_32x4d_best.pth
* unfreeze
Results
* 1:46:54.2
* 75.85 % accuracy
* save as: `13_se_resnext101_32x4d_best.pth`

### Exp12
* epoch 15
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* optim SGD
* output class: 5
* loaded model: 11_se_resnext101_32x4d_best.pth
* unfreeze
Results
* 1:46:55.7
* 74.51 % accuracy
* save as: `12_se_resnext101_32x4d_best.pth`

### Exp11
* epoch 10
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* optim Adam
* output class: 5
* freeze
Results
* 0:41:44.1
* 68.18 % accuracy
* save as: `11_se_resnext101_32x4d_best.pth`


## 8/21
### Change the image aug.


### Exp11 (ommited)
Parameters
* epoch 10
* batch_size 64 
* image_size 300
* model efficientnet-b0
* optim Adam
* freeze
Results
* 
*  % accuracy
* save as: `11_efficientnet-b0_best.pth`

### Exp10
Parameters
* epoch 15
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* optim SGD
* load_model_para 8_se_resnext101_32x4d_best.pth
* unfreeze
Results
* 1:48:12.1
* 66.06 % accuracy
* save as: `10_se_resnext101_32x4d_best.pth`


### Exp8
Parameters
* epoch 15
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* optim SGD
* load_model_para 7_se_resnext101_32x4d_best.pth
* unfreeze
Results
* 1:46:16.5
* 66.91 % accuracy
* save as: `8_se_resnext101_32x4d_best.pth`

## 8/20
### Check age distribution
Diagram outputed.

### Exp6
Parameters
* epoch 5 
* batch_size 64 
* image_size 300
* model se_resnext101_32x4d
* unfreeze
Results
* 0:36:17.7
* 64.4 % accuracy

### Exp5
Parameters
* epoch 5 
* batch_size 64 
* image_size 330
* model se_resnext101_32x4d
* unfreeze
Results
* 31 mincd 
* 63.52 % accuracy
* min/epoch


### Exp4
Parameters
* epoch 5 
* batch_size 64 
* image_size 330
* model se_resnext101_32x4d
* freeze
Results
* 0:23:54.4
* 56.95 % accuracy
* 4 min/epoch

### Exp2
Parameters
* epoch 5 
* batch_size 64 
* image_size 330
* model se_resnext101_32x4d
Results
* 39 min
* best 62.51 % accuracy
* 8 min/epoch

## 8/19
### Exp1
Parameters
* epoch 10 
* batch_size 64 
* image_size 1024 
* else default
Results
* 2.25 hr
* 64.91 % accuracy