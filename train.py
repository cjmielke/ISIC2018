import argparse
import os
import uuid

from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

import segmentation_models as sm
sm.set_framework('tf.keras')

from augmentation import get_validation_augmentation, get_preprocessing, get_training_augmentation
from callbacks import buildMetricCallback, MyCSVLogger
from generators import Dataset, Dataloder
from segmentation_models.losses import dice_loss, jaccard_loss, binary_focal_loss, categorical_crossentropy, \
    categorical_focal_loss, binary_crossentropy, bce_dice_loss, cce_dice_loss, binary_focal_dice_loss, \
    categorical_focal_dice_loss, bce_jaccard_loss, cce_jaccard_loss, binary_focal_jaccard_loss, \
    categorical_focal_jaccard_loss

from settings import trainingImages, trainingMasks, validationImages, validationMasks, WIDTH, HEIGHT

# For later hyperparameter tuning fun!
def getLoss(lossName):
    losses = dict(jaccard_loss=jaccard_loss, dice_loss=dice_loss, binary_focal_loss=binary_focal_loss,
                  categorical_focal_loss=categorical_focal_loss, binary_crossentropy=binary_crossentropy,
                  categorical_crossentropy=categorical_crossentropy,
                  bce_dice_loss=bce_dice_loss, bce_jaccard_loss=bce_jaccard_loss,
                  cce_dice_loss=cce_dice_loss, cce_jaccard_loss=cce_jaccard_loss,
                  binary_focal_dice_loss=binary_focal_dice_loss, binary_focal_jaccard_loss=binary_focal_jaccard_loss,
                  categorical_focal_dice_loss=categorical_focal_dice_loss, categorical_focal_jaccard_loss=categorical_focal_jaccard_loss)

    if lossName not in losses: lossName += '_loss'          # in case I get lazy at the commandline
    try:
        return losses[lossName]
    except:
        raise ValueError(f'Loss {lossName} is not valid. Pick one from {losses.keys()} ')


def trainModel(args, learningRate=0.0001, EPOCHS=10):

    preprocess_input = sm.get_preprocessing(args.backbone)

    modelName = args.model.lower()
    if modelName=='unet':
        model = sm.Unet(args.backbone, classes=1, activation='sigmoid')
    elif modelName == 'pspnet':
        model = sm.PSPNet(args.backbone, classes=1, activation='sigmoid')
    elif modelName=='fpn':
        model = sm.FPN(args.backbone, classes=1, activation='sigmoid')
    elif modelName=='linknet':
        model = sm.Linknet(args.backbone, classes=1, activation='sigmoid')
    else: raise ValueError('Not a valid model type')

    model.summary()         # TODO - get graphviz plots for later to understand how the backbones are hacked

    optim = keras.optimizers.Adam(learningRate)
    loss = getLoss(args.lossfunc)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim, loss, metrics)

    train_dataset = Dataset(
        trainingImages, trainingMasks, paddingShape=(WIDTH, HEIGHT),
        augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocess_input)
    )
    valid_dataset = Dataset(
        validationImages, validationMasks, paddingShape=(WIDTH, HEIGHT),
        augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=args.batchsize, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # some sanity checks .... still getting the hang of Keras's new style data loader
    assert train_dataloader[0][0].shape == (args.batchsize, HEIGHT, WIDTH, 3)
    assert train_dataloader[0][1].shape == (args.batchsize, HEIGHT, WIDTH, 1)

    # I'll get Weights and Biases working on the new server soon, but for now, I'll just log experiments to flatfiles
    runID = str(uuid.uuid4())
    logger = MyCSVLogger(f'logs/experiment-{runID}.csv')    # super hacky quick technique to log hyperparams alongside training curves
    logger.hyperparams = args
    logger.otherfields = dict(nparams=model.count_params())
    callbacks = [
        ModelCheckpoint(f'checkpoints/{runID}_best_model.h5', save_weights_only=True, save_best_only=True, mode='min', verbose=1),
        keras.callbacks.ReduceLROnPlateau(), logger
    ]

    # finally time to train!
    model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        #steps_per_epoch=10,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
        #validation_steps=1
    )


def get_args(bs=4, act='relu', fc1=128, fc2=64):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('-backbone', default='efficientnetb3', type=str)
        parser.add_argument('-model', default='Unet', type=str, help='One of Unet, PSPNet, Linknet, FPN')
        parser.add_argument('-batchsize', default=bs, type=int)
        parser.add_argument('-lossfunc', default='dice', type=str)
        parser.add_argument('-gpu', default=0, type=int)

        # params from prior projects that I might consider hyperparameter tuning with
        #parser.add_argument('-opt', default='adam', type=str)
        #parser.add_argument('-size', default=1024, type=int)
        #parser.add_argument('-dropout', default=0.0, type=float)
        #parser.add_argument('-weights', default=None, type=str)

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)          # makes my life a little easier
    trainModel(args)

