import numpy as np
from numpy.random import uniform

from constants import *


def next_batch():
    """
    Modify this function to ingest your data and returns it.
    :return: (inputs, targets). Could be a python generator.
    """
    x = np.array(uniform(size=(1, FULL_SEQUENCE_LENGTH, 1)), dtype='float32')
    # y = [np.mean(y) for y in [x[:, i - SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, x.shape[1])]]
    y = x[:, FULL_SEQUENCE_LENGTH - SEQUENCE_LENGTH + 1:]
    return np.array(x, dtype='float32'), np.array(np.reshape(y, (1, -1, 1)), dtype='float32')
