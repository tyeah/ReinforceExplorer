import tensorflow as tf
from tensorflow.contrib import slim

# function approximators: v(s) or pi(s)
class Estimator(object):
    def __init__(self, fn_name):
        self.reuse = False
        estimators = {
                'cnn': cnn,
                'fc': fc,
                'fc_action': fc_action
                }
        self.est_fn = estimators[fn_name]

    def get_estimator(self, inputs, **kwargs):
        if not self.reuse:
            self.reuse = False
        if len(inputs) == 1:
            ret = self.est_fn(inputs[0], reuse=self.reuse, **kwargs)
        else:
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
