import os.path
import h5py

from . import common


BOTTLENECK_FILE = os.path.join(common.TOP, 'bottleneck.hdf5')


def get_bottleneck_file(input_x, input_y, mode='r'):
    filename = f'bottleneck.{input_x}x{input_y}.hdf5'
    path = os.path.join(common.TOP, filename)
    return h5py.File(path, mode)
