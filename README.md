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


### Next steps
- [ ] Implement self-supervised contrastive loss on derm images to pre-train backbones and perform domain transfer learning.
    - Question : does this pretraining improve performance?
