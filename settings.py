'''
WIDTH, HEIGHT = 1024, 768

trainingImages = '../data/training_resized'
trainingMasks = '../data/training_masks_resized'

validationImages = '../data/validation_resized'
validationMasks = '../data/validation_masks_resized'
'''


# After early experiments, it became clear that I should just focus on low-res for now.
# Once the server is finished I can do much larger images.

WIDTH, HEIGHT = 512, 512

trainingImages = '../data/resized_512/training'
trainingMasks = '../data/resized_512/training_masks'

validationImages = '../data/resized_512/validation'
validationMasks = '../data/resized_512/validation_masks'



