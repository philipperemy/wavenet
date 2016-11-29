import numpy as np
from numpy.random import uniform

from constants import *


def next_batch():
    """
    Modify this function to ingest your data and returns it.
    :return: (inputs, targets). Could be a python generator.
    """
    x = np.expand_dims(uniform(size=FULL_SEQUENCE_LENGTH), axis=1)
    y = np.expand_dims(np.array([np.mean(y) for y in [x[i - SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(x))]]), axis=1)
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')
