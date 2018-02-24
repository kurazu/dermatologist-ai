import sys
import os.path
from glob import glob
from multiprocessing import Pool
import math
import functools

from keras.preprocessing import image
import numpy as np
from tqdm import tqdm

from . import common
from . import utils
from . import storage


DATA_DIR = os.path.join(common.TOP, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')


@utils.to_list
def load_dataset(path):
    class_names = os.listdir(path)
    for class_name in class_names:
        for file_path in glob(os.path.join(path, class_name, '*.jpg')):
            yield file_path, class_name


DATASETS = [
    ('train', TRAIN_DIR),
    ('test', TEST_DIR),
    ('valid', VALID_DIR)
]


def get_mask(n, one_idx):
    result = np.zeros(n)
    result[one_idx] = 1.0
    return result


CLASS_MAPPING = {
    name: get_mask(len(common.CLASSES), i)
    for i, name in enumerate(common.CLASSES)
}


def load_img_array(input_x, input_y, path):
    img = image.load_img(path, target_size=(input_x, input_y))
    img_arr = image.img_to_array(img, data_format='channels_last')
    return img_arr


def store_bottleneck_features(
    f, pretrained_model, input_x, input_y, name, dataset
):
    batch_size = 32
    workers = 8
    features_shape = (len(dataset), *pretrained_model.output_shape[1:])
    features_dset = f.create_dataset(
        f'{name}_features', features_shape, dtype='f'
    )
    labels_dset = f.create_dataset(
        f'{name}_labels', (len(dataset), len(common.CLASSES)), dtype='f'
    )
    batches = utils.grouper(dataset, batch_size)
    total_batches = math.ceil(len(dataset) / batch_size)
    load_img_array_callback = functools.partial(
        load_img_array, input_x, input_y
    )
    with Pool(workers) as pool:
        for batch_idx, batch in enumerate(tqdm(batches, total=total_batches)):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(batch)

            filenames = [path for path, _ in batch]
            img_arrays = pool.map(load_img_array_callback, filenames)
            img_tensors = np.stack(img_arrays)
            predictions = pretrained_model.predict(img_tensors)
            features_dset[start_idx:end_idx] = predictions

            class_names = (class_name for _, class_name in batch)
            labels = np.stack([
                CLASS_MAPPING[class_name] for class_name in class_names
            ])
            labels_dset[start_idx:end_idx] = labels


def main():
    input_x, input_y = map(int, sys.argv[1:])
    pretrained_model = common.get_pretrained_model(input_x, input_y)

    with storage.get_bottleneck_file(input_x, input_y, 'w') as f:
        for name, path in DATASETS:
            dataset = load_dataset(path)
            store_bottleneck_features(
                f, pretrained_model, input_x, input_y, name, dataset
            )


if __name__ == '__main__':
    main()
