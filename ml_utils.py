import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)


def count_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def by_name(tensor_name, index=0):
    return tf.get_default_graph().get_tensor_by_name(tensor_name + ':' + repr(index))


def create_convolution_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def _to_dilated_sequences(value, dilation):
    with tf.name_scope('to_dilated_sequences'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [pad_elements, 0], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def _revert_from_dilated_sequences(value, dilation):
    with tf.name_scope('revert_from_dilated_sequences'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])


def dilated_convolution(value, filter_, dilation, name='dilated_conv'):
    with tf.name_scope(name):
        dilated_sequences = _to_dilated_sequences(value, dilation)
        conv_sequences = tf.nn.conv1d(dilated_sequences, filter_, stride=1, padding='VALID')  # we use batch here.
        conv = _revert_from_dilated_sequences(conv_sequences, dilation)
        pad_conv = tf.pad(conv, [[0, 0], [dilation, 0], [0, 0]])
        return pad_conv


def causal_convolution(value, filter_, name='causal_convolution'):
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        filter_width = tf.shape(filter_)[0]  # should be 2 here.
        # Pads n zeros before, for dimension D where n = arr[D, 0]
        # Pads n zeros after, for dimension D where n = arr[D, 1]
        padding = [[0, 0], [(filter_width - 1), 0], [0, 0]]
        # here we pad on axis=1, with 1 line of zeros.
        padded = tf.pad(value, padding)
        # convolve and output (batch, width, output_channels)
        # for example if values.shape = (1, 10, 1), output.shape = (1, 10, 5).
        # filter width = 2, stride = 1, overlaps when we move the filters.
        conv_values = tf.nn.conv1d(padded, filter_, stride=1, padding='VALID')
        return conv_values
