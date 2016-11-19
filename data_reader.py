import numpy as np
from numpy.random import uniform


def next_batch():
    x = np.expand_dims(uniform(size=32), axis=1)
    y = np.expand_dims(np.expand_dims(np.mean(x), axis=0), axis=0)  # should be close to 0.5
    return x, y


if __name__ == '__main__':
    x, y = next_batch()
    print(x.shape)
    print(y.shape)
