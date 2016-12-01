from wavenet.ml_utils import *


class WaveNet(object):
    def __init__(self, dilations, sequence_length, x_placeholder, y_placeholder, use_biases=False):
        self.dilations = dilations
        self.sequence_length = sequence_length
        self.residual_channels = 16  # Not specified in the paper.
        self.dilation_channels = 32  # Not specified in the paper.
        self.skip_channels = 16  # Not specified in the paper.
        self.filter_width = 2  # Convolutions just use 2 samples. This parameter should not be changed.
        self.initial_channels = 1
        self.use_biases = use_biases
        self.variables = self._create_variables()
        self.batch_size = 1
        self.predict_func = self._init_predict_tensor(x_placeholder)
        self.loss_func = self._init_loss_tensor(y_placeholder)

    def pred(self):
        return self.predict_func

    def loss(self):
        return self.loss_func

    def _init_predict_tensor(self, x):
        x = tf.reshape(tf.cast(x, tf.float32), [self.batch_size, -1, 1])
        out = self._create_network(x)
        return tf.identity(out, name='prediction')

    def _init_loss_tensor(self, y, name='loss'):
        with tf.name_scope(name):
            out = self.pred()
            slice_size = tf.shape(out)[1] - self.sequence_length
            out = tf.reshape(tf.slice(tf.reshape(out, [-1]), begin=[self.sequence_length - 1], size=[slice_size]),
                             [-1, 1])
            reduced_loss = tf.reduce_sum(tf.square(tf.sub(out, y)))
            return reduced_loss

    def _create_network(self, inputs):
        skip_connections = []
        out_layer = self._create_causal_layer(inputs)
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer_{}'.format(layer_index)):
                    skip, out_layer = self._create_dilated_layer(out_layer, layer_index, dilation)
                    skip_connections.append(skip)
        with tf.name_scope('post_processing'):
            skips = tf.add_n(skip_connections)
            skips = tf.nn.relu(skips)
            w_skip_dense_1 = self.variables['skip']['skip_1']
            w_skip_dense_2 = self.variables['skip']['skip_2']
            skips = tf.nn.conv1d(skips, w_skip_dense_1, stride=1, padding='VALID')
            if self.use_biases:
                skips = tf.add(skips, self.variables['skip']['skip_1_bias'])
            skips = tf.nn.relu(skips)
            skips = tf.nn.conv1d(skips, w_skip_dense_2, stride=1, padding='VALID')
            if self.use_biases:
                skips = tf.add(skips, self.variables['skip']['skip_2_bias'])
        return skips

    def _create_causal_layer(self, inputs):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_convolution(inputs, weights_filter)

    def _create_dilated_layer(self, inputs, layer_index, dilation):
        with tf.name_scope('dilated_layer'):
            variables = self.variables['dilated_stack'][layer_index]
            w_f = variables['filter']
            w_g = variables['gate']

            conv_filter = dilated_convolution(inputs, w_f, dilation)
            conv_gate = dilated_convolution(inputs, w_g, dilation)

            if self.use_biases:
                conv_filter = tf.add(conv_filter, variables['filter_bias'])
                conv_gate = tf.add(conv_gate, variables['gate_bias'])

            z = tf.mul(tf.tanh(conv_filter), tf.sigmoid(conv_gate))

            w_o = variables['dense']
            conv_o = tf.nn.conv1d(z, w_o, stride=1, padding='VALID')

            w_s = variables['skip']
            conv_s = tf.nn.conv1d(z, w_s, stride=1, padding='VALID')

            if self.use_biases:
                conv_o = tf.add(conv_o, variables['dense_bias'])
                conv_s = tf.add(conv_s, variables['skip_bias'])

            return conv_s, inputs + conv_o

    def _create_variables(self):
        var = dict()
        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                layer['filter'] = create_convolution_variable('filter', [self.filter_width,
                                                                         1,  # in_channels = 1
                                                                         self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer_{}'.format(i)):
                        current = dict()
                        current['filter'] = create_convolution_variable('filter', [self.filter_width,
                                                                                   self.residual_channels,
                                                                                   self.dilation_channels])
                        current['gate'] = create_convolution_variable('gate', [self.filter_width,
                                                                               self.residual_channels,
                                                                               self.dilation_channels])
                        # 1 x 1 => will be the output of the layer.
                        current['dense'] = create_convolution_variable('dense', [1,
                                                                                 self.dilation_channels,
                                                                                 self.residual_channels])
                        # 1 x 1 => will be sent to the skip output.
                        current['skip'] = create_convolution_variable('skip', [1,
                                                                               self.dilation_channels,
                                                                               self.skip_channels])
                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable('filter_bias', [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable('gate_bias', [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable('dense_bias', [self.residual_channels])
                            current['skip_bias'] = create_bias_variable('skip_bias', [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('skip'):
                skip = dict()
                # 1 x 1
                skip['skip_1'] = create_convolution_variable('skip_1', [1, self.skip_channels, self.skip_channels])
                # 1 x 1
                skip['skip_2'] = create_convolution_variable('skip_2', [1, self.skip_channels, 1])  # in_channels = 1

                if self.use_biases:
                    skip['skip_1_bias'] = create_bias_variable('skip_1_bias', [self.skip_channels])
                    skip['skip_2_bias'] = create_bias_variable('skip_2_bias', [1])

                var['skip'] = skip
        return var
