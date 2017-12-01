"""
This script implements data preprocessing and other data-related operations.
"""

import common
import numpy as np
# from align_dlib import AlignDlib
from keras import backend

np.random.seed(4423)
backend.set_image_dim_ordering('th')


"""
Data operations.
"""

def load_data(filename, reshape=False):
    """
    Load X data from the specified file. Reshape it to 2D if necessary.
    """
    data = np.load(filename)
    if reshape:
        data = common.reshape(data)
    return data


def load_label(filename):
    """
    Load y data from the given file.
    """
    data = np.load(filename)
    data = np.ravel(np.reshape(data, (len(data), 1)))
    return data


"""
Data pre-processing and augmentation.
"""

def augment(data, label):
    """
    Augments data with the corresponding horizontally-flipped images. Also
    augments the labels.
    """
    data_aug = np.copy(data)
    for i, image in enumerate(data_aug):
        data_aug[i][0] = np.fliplr(image[0])

    return (np.vstack((data, data_aug)), np.append(label, label))


def align_face(data):
    """
    Re-aligns the face given in the data matrix.
    # TODO
    """
    ad = AlignDlib(face_predictor)
