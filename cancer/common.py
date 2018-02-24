import os.path
from keras.applications.inception_resnet_v2 import InceptionResNetV2

CLASSES = [
    'melanoma',
    'nevus',
    'seborrheic_keratosis'
]

HERE = os.path.dirname(__file__)
TOP = os.path.join(HERE, '..')
BOTTLENECK_FILE = os.path.join(TOP, 'bottleneck.hdf5')
IMAGE_INPUT_SIZE = 512, 512


def get_pretrained_model():
    pretrained_model = InceptionResNetV2(
        include_top=False, weights='imagenet',
        input_shape=(*IMAGE_INPUT_SIZE, 3), pooling=None
    )
    pretrained_model.summary()
    pretrained_model._make_predict_function()
    return pretrained_model
