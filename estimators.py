import tensorflow as tf
from tensorflow.contrib import slim


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, out_shape, **kwargs):
        self.kwargs = kwargs
        self.reuse = False
        self.out_shape = out_shape
        self.trainable = kwargs.get('trainable', True)

    @property
    def state_size(self):
        return self.out_shape
    @property
    def output_size(self):
        return self.out_shape

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            #assert self.inputs_shape == tuple(inputs.get_shape().as_list()[2:])
            with slim.arg_scope([slim.conv2d], 
                    num_outputs=self.kwargs.get('num_outputs', self.out_shape[-1]),
                    kernel_size=self.kwargs.get('kernel_size', 1),
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    biases_initializer=None, # zero bias
                    trainable=self.trainable):
                state_ = tf.reshape(state, (-1,) + tuple(self.out_shape))
                z = tf.nn.sigmoid(slim.conv2d(inputs) + slim.conv2d(state_))
                r = tf.nn.sigmoid(slim.conv2d(inputs) + slim.conv2d(state_))
                hc = tf.nn.tanh(slim.conv2d(inputs) + slim.conv2d(r * state_))
                h = (1 - z) * state_ + z * hc
                #output = tf.flatten(h)
                output = h
                if not self.reuse:
                    self.reuse = True
                return output, output

def rnn_preprocess(inputs, num_out, trainable, reuse, scope=None, **kwargs):

  num_features = kwargs['num_features']
  input_shape = tf.shape(inputs)
  batch_size = input_shape[0]
  num_variables = input_shape[2]
  if scope == None: scope = 'rnn'
  with tf.variable_scope(scope, reuse=resue):
    gru = ConvGRUCell([num_variables, num_out], trainable=trainable)
    inputs = tf.rehsape(inputs, [num_features, batch_size, num_variables, -1])
    inputs = tf.unpack(inputs)
    outputs, states = tf.nn.rnn[gru, inputs, dtype = tf.float32)

    output = tf.reshape(outputs[-1], [batch_size, num_variables, num_out])

  return output

# function approximators: v(s) or pi(s)
class Estimator(object):
    def __init__(self, fn_name):
        self.reuse = False
        estimators = {
                'cnn': cnn,
                'fc': fc,
                'fc_action': fc_action,
                'parallel': parallel,
                'parallel_action': parallel_action,
                }
        self.est_fn = estimators[fn_name]

    def get_estimator(self, inputs, **kwargs):
        if not self.reuse:
            self.reuse = False
        if len(inputs) == 1:
            ret = self.est_fn(inputs[0], reuse=self.reuse, **kwargs)
        else:
            if kwargs['rnn_preprocess']:
              inputs = rnn_preprocess(inputs, kwargs['num_out'], True, reuse, scope='rnn', **kwargs)
            ret = self.est_fn(inputs, reuse=self.reuse, **kwargs)
        self.reuse = True
        return ret

       
def cnn(inputs, num_out, num_cnn_layers, num_fc_layers, reuse, trainable, scope=None, **kwargs):
    net = inputs
    if scope == None: scope = 'cnn'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                trainable=trainable,
                activation_fn=tf.nn.relu):
            for nl in xrange(num_cnn_layers):
                net = slim.conv2d(net, 32, [3, 3], reuse=reuse, scope='conv%d' % (nl + 1))
                net = slim.max_pool2d(net, 2)
            net = slim.flatten(net)
            for nl in xrange(num_fc_layers - 1):
                fc_num_out = net.get_shape().as_list()[-1]
                assert fc_num_out >= 2
                fc_num_out = int(fc_num_out / 2)
                net = slim.fully_connected(net, fc_num_out, scope='fc_out%d' % (nl + 1))
            net = slim.fully_connected(slim.flatten(net), num_out, scope='fc_out_2', activation_fn=None)
            return net

def fc(inputs, num_out, num_hids, reuse, trainable, scope=None, **kwargs):
    net = inputs
    if scope == None: scope = 'fc'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                trainable=trainable,
                activation_fn=tf.nn.tanh):
            net = slim.flatten(net)
            for nl, nh in enumerate(num_hids):
                net = slim.fully_connected(net, nh, scope='fc_out%d' % (nl + 1))
            net = slim.fully_connected(slim.flatten(net), num_out, scope='fc_out_2', activation_fn=None)
            return net

def fc_action(inputs, actions, num_out, num_hids, reuse, trainable, scope=None, **kwargs):
    net = inputs
    merge_layer = kwargs.get('merge_layer', -1)
    if scope == None: scope = 'fc'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                trainable=trainable,
                activation_fn=tf.nn.tanh):
            net = slim.flatten(net)
            for nl, nh in enumerate(num_hids):
                net = slim.fully_connected(net, nh, scope='fc_out%d' % (nl + 1))
                '''
                if nl == len(num_hids) + merge_layer + 1:
                    print 'concat'
                    net = tf.concat(1, (net, actions))
                '''
            if len(actions.get_shape()) == 2:
                net = tf.concat(1, (net, tf.cast(actions, net.dtype)))
            elif len(actions.get_shape()) == 1:
                net = tf.concat(1, (net, tf.cast(tf.expand_dims(actions, 1), net.dtype)))
            net = slim.fully_connected(slim.flatten(net), num_out, scope='fc_out_2', activation_fn=None)
            return net

def parallel(inputs, reuse, trainable, scope=None, **kwargs):
    net = inputs
    num_features = kwargs['num_features']
    net = tf.reshape(net, (tf.shape(net)[0], num_features, -1, 1))
    if scope == None: scope = 'parallel'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                trainable=trainable,
                padding='VALID',
                activation_fn=tf.nn.relu):
            for nl, nh in enumerate(kwargs['num_hids']):
                net = slim.conv2d(net, nh, [num_features if nl == 0 else 1, 1], reuse=reuse, scope='conv%d' % (nl + 1))
            net = slim.conv2d(net, 1, [1, 1], reuse=reuse, activation_fn=None, scope='fc')
            net = tf.reshape(net, (tf.shape(net)[0], 1, -1))
            return net

def parallel_action(inputs, actions, reuse, trainable, scope=None, **kwargs):
    net = inputs
    num_features = kwargs['num_features']
    net = tf.reshape(net, (tf.shape(net)[0], num_features, -1, 1))
    if scope == None: scope = 'parallel'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                trainable=trainable,
                padding='VALID',
                activation_fn=tf.nn.relu):
            for nl, nh in enumerate(kwargs['num_hids']):
                net = slim.conv2d(net, nh, [num_features if nl == 0 else 1, 1], reuse=reuse, scope='conv%d' % (nl + 1))
            net = tf.concat(3, (net, tf.cast(tf.reshape(actions, (tf.shape(net)[0], 1, -1, 1)), net.dtype)))
            net = slim.conv2d(net, 1, [1, 1], reuse=reuse, activation_fn=None, scope='fc')
            net = tf.reshape(net, (tf.shape(net)[0], 1, -1))
            return net
