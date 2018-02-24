from keras.preprocessing import image
import os.path
from glob import glob
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import math

from . import common
from . import utils


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


def load_img_array(path):
    img = image.load_img(path, target_size=common.IMAGE_INPUT_SIZE)
    img_arr = image.img_to_array(img, data_format='channels_last')
    return img_arr


def store_bottleneck_features(f, pretrained_model, name, dataset):
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
    with Pool(workers) as pool:
        for batch_idx, batch in enumerate(tqdm(batches, total=total_batches)):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(batch)

            filenames = [path for path, _ in batch]
            img_arrays = pool.map(load_img_array, filenames)
            img_tensors = np.stack(img_arrays)
            predictions = pretrained_model.predict(img_tensors)
            features_dset[start_idx:end_idx] = predictions

            class_names = (class_name for _, class_name in batch)
            labels = np.stack([
                CLASS_MAPPING[class_name] for class_name in class_names
            ])
            labels_dset[start_idx:end_idx] = labels


def main():
    pretrained_model = common.get_pretrained_model()

    with h5py.File(common.BOTTLENECK_FILE, 'w') as f:
        for name, path in DATASETS:
            dataset = load_dataset(path)
            store_bottleneck_features(f, pretrained_model, name, dataset)


if __name__ == '__main__':
    main()
