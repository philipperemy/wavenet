import collections
import json

import numpy as np

from data_reader import next_batch
from file_logger import FileLogger
from ml_utils import *
from wavenet import WaveNet

LEARNING_RATE = 1e-3
WAVENET_PARAMS = 'wavenet_params.json'
MOMENTUM = 0.9
SEQUENCE_LENGTH = 32


def main():
    with open(WAVENET_PARAMS, 'r') as f:
        wavenet_params = json.load(f)

    with tf.name_scope('create_inputs'):
        batch_x = tf.placeholder('float32', [SEQUENCE_LENGTH, 1])
        batch_y = tf.placeholder('float32', [1, 1])

    net = WaveNet(wavenet_params['dilations'], SEQUENCE_LENGTH)
    loss = net.loss(batch_x, batch_y)
    pred = by_name('loss/prediction')
    optimizer = create_adam_optimizer(LEARNING_RATE, MOMENTUM)
    trainable = tf.trainable_variables()
    grad_update = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Total # of parameters to train: {}'.format(count_trainable_parameters()))
    file_logger = FileLogger('hello.txt', ['step', 'training_loss'])
    d = collections.deque(maxlen=10)
    for step in range(1, int(1e9)):
        x, y = next_batch()
        loss_value, _, pred_value = sess.run([loss, grad_update, pred],
                                             feed_dict={batch_x: x,
                                                        batch_y: y})
        print('y = {}, p = {}'.format(y.flatten(), pred_value))
        d.append(loss_value)
        mean_loss = np.mean(d)
        file_logger.write([step, mean_loss])
        print('loss = {:.6f}'.format(mean_loss))
    file_logger.close()


if __name__ == '__main__':
    main()
