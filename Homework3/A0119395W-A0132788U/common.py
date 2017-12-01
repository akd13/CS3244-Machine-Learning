"""
This script defines all common parameters and functions shared across files.
"""
import csv
import numpy as np
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


"""
Hyper-parameters.
"""
N_FOLDS = 5

# For dimensionality reduction
DIMENSION = 400

# Common parameters for neural networks
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 40

# Input image dimensions
IMG_ROWS, IMG_COLS = 50, 37
INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)

# Data files
X_train_file = 'X_train.npy'
X_test_file = 'X_test.npy'
y_train_file = 'y_train.npy'

# After augmentation
X_train_aug = 'X_train_aug.npy'
y_train_aug = 'y_train_aug.npy'

# After alignment
X_train_final = 'X_train_final.npy'
y_train_final = 'y_train_final.npy'


"""
Common, utility functions.
"""

def save_results(filename, predicted):
    """
    Saves results of predictions to the specified file location, following the
    required format in the competition.
    """
    image_ids = range(len(predicted))
    with open(filename, 'w', newline='') as f:
        wr = csv.writer(f, delimiter=',')
        wr.writerow(('ImageId', 'PredictedClass'))
        wr.writerows(zip(image_ids, predicted))


def reduce_dimension(X_input, dim):
    """
    Performs PCA dimensionality reduction on the given input to
    reduce it to the desired dimension. Returns the transformed data.
    """
    pca = PCA(n_components=dim)
    return pca.fit_transform(X_input)


def normalize(data):
    """
    Normalization of the data by transforming it into a Gaussian distribution.
    """
    scaler = preprocessing.StandardScaler().fit(data)
    scaler.transform(data) 
    return data


def normalize_CNN(data):
    """
    Normalize data to the range 0-1 for the CNN models. Somehow normalising it
    to Gaussian distribution doesn't work, since the models wouldn't learn
    anything..
    """
    return data / 255.0

 
def reshape(data):
    """
    Reshapes one-dimensional image data to 2D.
    """
    return data.reshape(data.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]).astype('float32')


def show_image(image):
    """
    Helper function to show the image in gray scale using pyplot.
    """
    pyplot.imshow(image, cmap=pyplot.cm.gray)
    pyplot.axis('off')
    pyplot.show()



def majority_vote(predictions):
    """
    Given a list of predictions, return the majority vote of these predictions.
    """
    result = []
    n = len(predictions[0])

    for i in range(n):
        i_preds = []
        for p in predictions:
            i_preds.append(p[i])

        median = int(np.median(i_preds))
        result.append(median)

    return result
