import os.path
from keras.applications.inception_resnet_v2 import InceptionResNetV2

CLASSES = [
    'melanoma',
    'nevus',
    'seborrheic_keratosis'
]

HERE = os.path.dirname(__file__)
TOP = os.path.join(HERE, '..')


def get_pretrained_model(input_x, input_y):
    pretrained_model = InceptionResNetV2(
        include_top=False, weights='imagenet',
        input_shape=(input_x, input_y, 3), pooling=None
    )
    pretrained_model.summary()
    pretrained_model._make_predict_function()
    return pretrained_model
