from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
import os.path
from glob import glob
import h5py
import functools
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import math

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BOTTLENECK_FILE = os.path.join(HERE, 'bottleneck.hdf5')

IMAGE_INPUT_SIZE = 512, 512
NUMBER_OF_SAMPLES = 2754


def to_list(func):
    @functools.wraps(func)
    def to_list_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return list(result)
    return to_list_wrapper


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip(*args)


@to_list
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

CLASSES = [
    'melanoma',
    'nevus',
    'seborrheic_keratosis'
]


def get_mask(n, one_idx):
    result = np.zeros(n)
    result[one_idx] = 1.0
    return result


CLASS_MAPPING = {
    name: get_mask(len(CLASSES), i)
    for i, name in enumerate(CLASSES)
}


def load_img_array(path):
    img = image.load_img(path, target_size=IMAGE_INPUT_SIZE)
    img_arr = image.img_to_array(img, data_format='channels_last')
    return img_arr


def store_bottleneck_features(f, pretrained_model, name, dataset):
    batch_size = 32
    workers = 16
    features_shape = (len(dataset), *pretrained_model.output_shape[1:])
    features_dset = f.create_dataset(
        f'{name}_features', features_shape, dtype='f'
    )
    labels_dset = f.create_dataset(
        f'{name}_labels', (len(dataset), len(CLASSES)), dtype='f'
    )
    batches = grouper(dataset, batch_size)
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
    pretrained_model = InceptionResNetV2(
        include_top=False, weights='imagenet',
        input_shape=(*IMAGE_INPUT_SIZE, 3), pooling=None
    )
    pretrained_model.summary()
    pretrained_model._make_predict_function()

    with h5py.File(BOTTLENECK_FILE, 'w') as f:
        for name, path in DATASETS:
            dataset = load_dataset(path)
            store_bottleneck_features(f, pretrained_model, name, dataset)
    return
    import pdb; pdb.set_trace()

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_INPUT_SIZE,
        batch_size=32,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False
    )  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(generator, NUMBER_OF_SAMPLES)
# save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

if __name__ == '__main__':
    main()
