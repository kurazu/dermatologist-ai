import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping

from . import model
from . import storage


EPOCHS = 50
BATCH_SIZE = 32


def get_model_and_callbacks(input_x, input_y, weights_prefix):
    decision_model = model.get_compiled_model(
        input_x, input_y, dropout_rate=0.5
    )
    decision_model.summary()

    checkpointer = ModelCheckpoint(
        filepath=model.get_weights_file_path(input_x, input_y, weights_prefix),
        verbose=1, save_best_only=True
    )

    stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto'
    )
    callbacks = [checkpointer, stopping]
    return decision_model, callbacks


def train(
    input_x, input_y,
    train_features, train_labels,
    valid_features, valid_labels
):

    decision_model, callbacks = get_model_and_callbacks(
        input_x, input_y, 'bottle'
    )

    decision_model.fit(
        train_features, train_labels,
        validation_data=(valid_features, valid_labels),
        epochs=EPOCHS, batch_size=32,
        callbacks=callbacks, verbose=1,
        shuffle='batch'
    )


def main():
    input_x, input_y = map(int, sys.argv[1:])
    with storage.get_bottleneck_file(input_x, input_y) as f:
        train_features = f['train_features']
        train_labels = f['train_labels']
        valid_features = f['valid_features']
        valid_labels = f['valid_labels']
        train(
            input_x, input_y,
            train_features, train_labels,
            valid_features, valid_labels
        )


if __name__ == '__main__':
    main()
