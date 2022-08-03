# This is my first time ever playing with the Albumentations library and I LOVE IT
# These augmentation steps were borrowed from the keras segmentation library, however several steps were deprecated
# I also removed the random crop and make the images larger

import albumentations as A
from settings import HEIGHT, WIDTH

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, always_apply=True, border_mode=0),

        # this was cropping way too much - if I had more time I'd experiment with a 1024x768 crop on larger "resized" images
        # A.RandomCrop(height=320, width=320, always_apply=True),

        # A.IAAAdditiveGaussianNoise(p=0.2),            # deprecated
        A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False,
                                              p=0.2),
        # A.IAAPerspective(p=0.5),                      # deprecated
        #A.augmentations.geometric.transforms.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0,
        #                                                 mask_pad_val=0, fit_output=False, interpolation=1,
        #                                                 always_apply=False, p=0.5),

        A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0,
                                                         mask_pad_val=0, fit_output=False, interpolation=1,
                                                         always_apply=False, p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                # A.IAASharpen(p=1),
                A.augmentations.transforms.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=1.0),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        #A.Resize(HEIGHT-HEIGHT%6, WIDTH-WIDTH%6, interpolation=2, always_apply=True),           # PSPNet requires sizes divisible by 6
        A.Lambda(mask=round_clip_0_1),
        #A.ToFloat(always_apply=True)               # didn't fix my problem .... https://github.com/qubvel/segmentation_models/issues/509
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        A.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH),
        #A.ToFloat(always_apply=True)
        #A.Resize(HEIGHT-HEIGHT%6, WIDTH-WIDTH%6, interpolation=2, always_apply=True),
        #A.Lambda(mask=round_clip_0_1),
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


