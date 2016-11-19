import numpy as np

from ml_utils import *
from ml_utils import _to_dilated_sequences

np.set_printoptions(threshold=np.nan)

quantization_channels = 16
batch_size = 1


class MyTest(tf.test.TestCase):
    def test_dilated_convolution(self):
        with self.test_session():
            tf.set_random_seed(1)
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32')
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=2)
            x = tf.identity(x)
            print('x.shape (conv1d) = [batch, in_width, in_channels]')
            print('x.shape (conv2d) reshaped in = [batch, 1, in_width, in_channels]')
            print('x.shape = {}'.format(x.eval().shape))
            print('')

            num_filters = 5
            filter_width = 2
            w = tf.random_uniform([filter_width, 1, num_filters])
            print('w.shape = [filter_width, in_channels, out_channels]')
            print('w.shape = {}'.format(w.eval().shape))

            print(_to_dilated_sequences(x, dilation=4).eval())
            dilated_convolution(x, w, dilation=2)

    def test_causal_convolution(self):
        with self.test_session():
            tf.set_random_seed(1)

            # x.shape = [batch, in_width, in_channels]
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='float32')
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=2)
            x = tf.identity(x)
            print('x.shape = [batch, in_width, in_channels]')
            print('x.shape = {}'.format(x.eval().shape))
            print('')

            num_filters = 5
            filter_width = 2
            w = tf.random_uniform([filter_width, 1, num_filters])
            print('w.shape = [filter_width, in_channels, out_channels]')
            print('w.shape = {}'.format(w.eval().shape))
            conv_res = causal_convolution(x, w, name='causal_conv')
            print('output = {}'.format(conv_res.eval().shape))
            print(conv_res.eval())


if __name__ == '__main__':
    tf.test.main()
