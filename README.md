# ISIC2018

This repository makes use of the excellent [segmentation models](https://github.com/qubvel/segmentation_models) library (based on Keras) to segment skin lesions.
This library exposes 4 primary segmentation models, many pretrained backbone architectures (I count 34!), and a huge number of recommneded loss functions.

Overall, the flexibility of this library opens the door for tons of hyperparameter experiments to find the best architecutes.

### Overall process

1. Performed basic exploration of the dataset in [this notebook](EDA.ipynb)
2. Implemented training loop in pycharm
3. Executed training script in parallel on server with a variety of hyperparameter sweeps
    - [x] pretrained backbone architectures
    - Model types
      - [x] Unet
      - [ ] PSP
      - [ ] FPN
      - [ ] LinkNet
    - Loss functions
      - [x] Dice
      - [ ] .... many available
    - [ ] Learning rate optimization
    - [ ] Image size experiments (once server is finished)
4. Results analysis in [this notebook](ExpResults.ipynb)
   - Hyperparameter results
   - Visualized segmentation results on validation set for best models

### Results

IoU (Intersection over Union) of validation set images for various network backbones.

![boxplot](img/boxplot.png)

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
