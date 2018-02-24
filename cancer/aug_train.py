import sys

from keras.preprocessing.image import ImageDataGenerator

from . import bottleneck
from . import common
from . import train

BATCH_SIZE = 32
TRAIN_SAMPLES = 2000
VALIDATION_SAMPLES = 150


def pretrained_model_transformer(pretrained_model, data_generator):
    for images_batch, labels_batch in data_generator:
        input_batch = pretrained_model.predict(images_batch)
        yield input_batch, labels_batch


def main():
    input_x, input_y = map(int, sys.argv[1:])
    train_datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        bottleneck.TRAIN_DIR,
        target_size=(input_x, input_y),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        bottleneck.VALID_DIR,
        target_size=(input_x, input_y),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    pretrained_model = common.get_pretrained_model(input_x, input_y)

    transformed_train_generator = pretrained_model_transformer(
        pretrained_model, train_generator
    )
    transformed_validation_generator = pretrained_model_transformer(
        pretrained_model, validation_generator
    )

    decision_model, callbacks = train.get_model_and_callbacks(
        input_x, input_y, 'aug'
    )

    decision_model.fit_generator(
        transformed_train_generator,
        steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE,
        epochs=train.EPOCHS,
        validation_data=transformed_validation_generator,
        validation_steps=VALIDATION_SAMPLES // BATCH_SIZE,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
