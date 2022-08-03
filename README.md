# ISIC2018

This repository makes use of the excellent [segmentation models](https://github.com/qubvel/segmentation_models) library (based on Keras) to segment skin lesions.
This library exposes 4 primary segmentation models, many pretrained backbone architectures (I count 34!), and a huge number of recommneded loss functions.

Overall, the flexibility of this library opens the door for tons of hyperparameter experiments to find the best architectures, and since I just built a new server to capitalize on my apartments free electric service, this project served as a well-timed stress test!

### Overall process

1. Performed basic exploration of the dataset in [this notebook](EDA.ipynb)
2. Implemented training loop in pycharm
3. Executed training script in parallel on server with a variety of hyperparameter sweeps
    - [x] pretrained backbone architectures
    - Model types
      - [x] Unet
      - [ ] PSPNet ([Need to fix sizing problem](https://github.com/qubvel/segmentation_models/issues/404) )
      - [x] FPN
      - [x] LinkNet
    - Loss functions
      - [x] Dice
      - [x] .... many available
    - [ ] Learning rate optimization
    - [ ] Image size experiments (once server is finished)
4. Results analysis in [this notebook](ExpResults.ipynb)
   - Hyperparameter results
   - Visualized segmentation results on validation set for best models

### Results

##### Best models
IoU (Intersection over Union) of validation set images for various network backbones. The boxplot below shows the performance on the validation set for the last 5 epochs of training.

![boxplot](img/boxplot.png)

| backbone       | model   |   f1-score |   iou_score | lossfunc             |       loss |   val_loss |   val_iou_score |
|:---------------|:--------|-----------:|------------:|:---------------------|-----------:|-----------:|----------------:|
| efficientnetb1 | Unet    |   0.88234  |    0.798617 | binary_focal         | 0.00819351 |  0.0181599 |        0.785357 |
| efficientnetb2 | Unet    |   0.911623 |    0.842784 | dice                 | 0.0888625  |  0.131296  |        0.785293 |
| efficientnetb1 | FPN     |   0.905341 |    0.833141 | dice                 | 0.095059   |  0.137675  |        0.77449  |
| efficientnetb4 | Unet    |   0.909573 |    0.84113  | dice                 | 0.0907972  |  0.138601  |        0.770986 |
| efficientnetb1 | Unet    |   0.90641  |    0.835262 | dice                 | 0.0940188  |  0.140748  |        0.772056 |
| efficientnetb1 | Unet    |   0.908211 |    0.837725 | dice                 | 0.0923381  |  0.142192  |        0.769189 |
| efficientnetb3 | Unet    |   0.906731 |    0.836326 | dice                 | 0.0938347  |  0.14567   |        0.763504 |
| efficientnetb0 | Unet    |   0.905225 |    0.833252 | dice                 | 0.0951778  |  0.146715  |        0.760904 |
| densenet121    | Unet    |   0.893326 |    0.814384 | dice                 | 0.107119   |  0.149303  |        0.755145 |
| efficientnetb1 | Linknet |   0.908595 |    0.838424 | dice                 | 0.0917967  |  0.150779  |        0.757658 |
| efficientnetb5 | Unet    |   0.899602 |    0.828446 | dice                 | 0.100559   |  0.151404  |        0.751347 |
| efficientnetb7 | Unet    |   0.901872 |    0.831914 | dice                 | 0.0983607  |  0.151908  |        0.752823 |
| efficientnetb6 | Unet    |   0.901224 |    0.830108 | dice                 | 0.0989042  |  0.15686   |        0.744913 |
| inceptionv3    | Unet    |   0.888505 |    0.808403 | dice                 | 0.111858   |  0.163674  |        0.745207 |
| resnext50      | Unet    |   0.880391 |    0.796514 | dice                 | 0.119969   |  0.16466   |        0.735834 |
| efficientnetb1 | Unet    |   0.902192 |    0.82865  | dice                 | 0.0982751  |  0.165386  |        0.739244 |
| resnet18       | Unet    |   0.882507 |    0.798707 | dice                 | 0.117766   |  0.183193  |        0.717013 |
| vgg16          | Unet    |   0.849076 |    0.748702 | dice                 | 0.151741   |  0.187719  |        0.709424 |
| efficientnetb1 | Unet    |   0.90911  |    0.83909  | binary_crossentropy  | 0.0741291  |  0.19895   |        0.75581  |
| mobilenetv2    | Unet    |   0.888101 |    0.805828 | dice                 | 0.112328   |  0.2006    |        0.696462 |
| efficientnetb1 | Unet    |   0.903946 |    0.83181  | jaccard              | 0.168661   |  0.215668  |        0.78452  |
| efficientnetb1 | Unet    |   0.909267 |    0.839128 | binary_focal_dice    | 0.128996   |  0.227253  |        0.779946 |
| mobilenet      | Unet    |   0.890715 |    0.810495 | dice                 | 0.109713   |  0.243964  |        0.644045 |
| efficientnetb1 | Unet    |   0.911743 |    0.843169 | bce_dice             | 0.183988   |  0.393461  |        0.773411 |
| efficientnetb1 | Unet    |   0.910934 |    0.84266  | bce_jaccard          | 0.264161   |  0.411634  |        0.787613 |
| efficientnetb1 | Unet    |   0.908008 |    0.837967 | binary_focal_jaccard | 0.20995    |  0.440861  |        0.746803 |


##### Segmentations on validation data
Segmentations performed on a handful of validation images for the best-performing efficientnet model.

Lessons learned : I think one of the mistakes I made was using a mirrored padding approach in the augmentations pipeline. You can see that the borders of some images are flipped. Note the rulers in the 1st and 3rd image. In some cases, as in image 4, the flipping can produce regions that look like lesions! (heh, that rhymed). Training/inference with different padding settings might alleviate this. 

![seg](img/val_seg_3.png)
![seg](img/val_seg_4.png)
![seg](img/val_seg_1.png)
![seg](img/val_seg_2.png)
![seg](img/val_seg_5.png)
![seg](img/val_seg_6.png)
![seg](img/val_seg_7.png)
![seg](img/val_seg_8.png)
![seg](img/val_seg_9.png)




### Next steps
- [ ] Implement self-supervised contrastive loss on derm images to pre-train backbones and perform domain transfer learning.
    - Question : does this pretraining improve performance?
