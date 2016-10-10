import tensorflow as tf
from tensorflow.contrib import slim

# function approximators: v(s) or pi(s)
class Estimator(object):
    def __init__(self, fn_name):
        self.reuse = False
        estimators = {
                'cnn': cnn
                }
        self.est_fn = estimators[fn_name]

    def get_estimator(self, inputs, **kwargs):
        if not self.reuse:
            self.reuse = True
        if len(inputs) == 1:
            return self.est_fn(inputs[0], **kwargs)
        else:
            return self.est_fn(inputs, **kwargs)

def cnn(inputs, num_out, num_layers, reuse, trainable, scope=None):
    net = inputs
    if scope == None: scope = 'cnn'
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                trainable=trainable,
                activation_fn=tf.nn.relu):
            for nl in xrange(num_layers):
                net = slim.conv2d(net, 32, [3, 3], reuse=reuse, scope='conv%d' % (nl + 1))
                net = slim.max_pool2d(net, 2)
            net = slim.fully_connected(slim.flatten(net), num_out, scope='fc_out', activation_fn=None)
            return net

