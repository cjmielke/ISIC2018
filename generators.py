#import keras
from glob import glob

import cv2
from tensorflow import keras as keras
import os
import numpy as np

from settings import WIDTH, HEIGHT


# FIXME - LOL I didn't even need this - albumations does it better!
'''
def padding(img, yy, xx):

    h = img.shape[0]
    w = img.shape[1]
    #print(h, yy, w, xx)

    if h == yy and w == xx:
        return img

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    if img.shape[-1]==1:
        return np.pad(img, pad_width=((a, aa), (b, bb)), mode='constant')
    elif img.shape[-1]==3:
        img = np.stack(
            [np.pad(img[:, :, c], pad_width=((a, aa), (b, bb)),
            mode='constant', constant_values=0) for c in range(3)], axis=2
        )
        return img
    else:
        raise ValueError('Not a B&W or RGB image? I have no idea what to do here ...')
'''

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            paddingShape=None
    ):
        self.paddingShape = paddingShape
        #self.ids = os.listdir(images_dir)
        self.images = glob(os.path.join(images_dir, '*.jpg'))
        filenames = [os.path.split(fn)[1] for fn in self.images]
        masks = [fn.replace('.jpg', '_segmentation.png') for fn in filenames]
        self.masks = [os.path.join(masks_dir, fn) for fn in masks]

        # convert str names to class values on masks
        #self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def loadImg(self, fn):
        image = cv2.imread(fn)
        #if self.paddingShape:
        #    image = padding(image, self.paddingShape[0], self.paddingShape[1])
        return image#.astype('float32')

    def __getitem__(self, i):

        # read data
        image = self.loadImg(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.loadImg(self.masks[i])
        mask = mask[:, :, 0:1]
        mask[mask > 0] = 1
        if mask is None:
            raise ValueError(f"Couldn't load mask : {self.masks[i]}")

        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image, mask = image.astype('float32'), mask.astype('float32')
        return image, mask

    def __len__(self):
        return len(self.images)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


