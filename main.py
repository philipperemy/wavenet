import collections
import json

import numpy as np
from pprint import pprint
from constants import *
from data_reader import next_batch
from helpers import FileLogger
from wavenet import *

LEARNING_RATE = 1e-4
WAVENET_PARAMS = 'wavenet_params.json'
MOMENTUM = 0.9


def main():
    with open(WAVENET_PARAMS, 'r') as f:
        wavenet_params = json.load(f)
    pprint(wavenet_params)

    with tf.name_scope('create_inputs'):
        x_placeholder = tf.placeholder('float32', [FULL_SEQUENCE_LENGTH, 1])
        y_placeholder = tf.placeholder('float32', [FULL_SEQUENCE_LENGTH - SEQUENCE_LENGTH, 1])

    net = WaveNet(wavenet_params['dilations'], SEQUENCE_LENGTH, x_placeholder, y_placeholder, use_biases=True,
                  use_mean_loss=True)
    loss = net.loss()
    pred = net.pred()
    optimizer = create_adam_optimizer(LEARNING_RATE, MOMENTUM)
    trainable = tf.trainable_variables()
    grad_update = optimizer.minimize(loss, var_list=trainable)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    print('Total # of parameters to train: {}'.format(count_trainable_parameters()))
    file_logger = FileLogger('log.tsv', ['step', 'training_loss', 'benchmark_loss'])
    d = collections.deque(maxlen=10)
    benchmark_d = collections.deque(maxlen=10)
    for step in range(1, int(1e9)):
        x, y = next_batch()
        loss_value, _, pred_value = sess.run([loss, grad_update, pred],
                                             feed_dict={x_placeholder: x,
                                                        y_placeholder: y})
        # The mean converges to 0.5 for IID U(0,1) random variables. Good benchmark.
        benchmark_d.append(np.mean(np.square(0.5 - y)))
        d.append(loss_value)
        mean_loss = np.mean(d)
        benchmark_mean_loss = np.mean(benchmark_d)
        file_logger.write([step, mean_loss, benchmark_mean_loss])
        # print('y = {}, p = {}, mean_loss = {}, bench_loss = {}'.format(y, pred_value, mean_loss, benchmark_mean_loss))
    file_logger.close()


if __name__ == '__main__':
    main()
