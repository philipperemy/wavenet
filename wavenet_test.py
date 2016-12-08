import numpy as np

from wavenet import *
from wavenet.ml_utils import _to_dilated_sequences

np.set_printoptions(threshold=np.nan)


class WaveNetTests(tf.test.TestCase):
    def test_long_sequence(self):
        with self.test_session() as sess:
            full_sequence_length = 1024
            sequence_length = 32
            batch_x = tf.identity(np.expand_dims(np.array(range(0, full_sequence_length), dtype='float32'), axis=1))
            batch_y = tf.identity(
                np.expand_dims(np.array(range(0, full_sequence_length - sequence_length), dtype='float32'), axis=1))
            net = WaveNet([1, 2, 4, 8, 16], sequence_length, batch_x, batch_y)
            init = tf.initialize_all_variables()
            sess.run(init)
            net._init_predict_tensor(batch_x)
            net._init_loss_tensor(batch_y)

    def test_loss_2(self):
        with self.test_session() as sess:
            sequence_length = 4
            batch_x = tf.placeholder('float32', [sequence_length, 1])
            batch_y = tf.placeholder('float32', [1, 1])
            net = WaveNet([1, 2], sequence_length, batch_x, batch_y)
            loss = net.loss()
            p = net.pred()

            init = tf.initialize_all_variables()
            sess.run(init)

            loss_value_1, p1 = sess.run([loss, p], feed_dict={batch_x: np.array([[0], [1], [2], [3]]),
                                                              batch_y: np.array([[1]])})

            loss_value_2, p2 = sess.run([loss, p], feed_dict={batch_x: np.array([[0], [1], [2], [3]]),
                                                              batch_y: np.array([[0]])})

            assert loss_value_1 != loss_value_2
            assert p1 == p2

    def test_loss(self):
        with self.test_session() as sess:
            sequence_length = 4
            batch_x = tf.identity(np.array([[0], [1], [2], [3]], dtype='float32'))
            batch_y = tf.identity(np.array([[1]], dtype='float32'))
            net = WaveNet([1, 2], sequence_length, batch_x, batch_y)
            init = tf.initialize_all_variables()
            sess.run(init)
            p = net.pred()
            assert np.square(p.eval() - batch_y.eval()).flatten()[0] == net.loss().eval()

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
