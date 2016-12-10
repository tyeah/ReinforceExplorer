import tensorflow as tf
import numpy as np
from estimators import Estimator
import json, os

def main():
    config_file = "configs/ddpgcont_parallel_quad_func.json"
    max_iter = 1000
    #max_iter = 100
    bsize, num_vars = 40, 200
    max_x = 10


    config = json.load(open(config_file))
    save_file = 'weights/' + config_file.split('/')[-1].split('.')[0] + '/pretrain_%s_%s.ckpt' % (config['env_config']['action'], config['env_config']['state'])

    actor_config = config['estimator_params']['policy_network']
    dim_features = 1 if config['env_config']['state'] in ['gradient', 'variable'] else 2
    inputs = tf.placeholder(dtype=tf.float32, shape=(bsize, 
        actor_config['num_features'], num_vars, dim_features))
    actor = Estimator(config['estimator_params']
            ['policy_network']['name']).get_estimator(
            inputs=[tf.reshape(inputs, (bsize, dim_features, actor_config['num_features'] * num_vars))],
            scope='actor',
            **actor_config)
    if config['env_config']['action'] == 'step':
        loss = tf.reduce_sum(tf.square(tf.squeeze(actor) - inputs[:, -1, :, -1]))
    elif config['env_config']['action'] == 'coordinate_lr':
        loss = tf.reduce_sum(tf.square(tf.squeeze(actor) - config['env_config']['base_lr']))
    elif self.config['env_config']['action'] == 'params':
        if self.config['env_config']['opt_method'] == 'rmsprop':
            self.num_actions = 4
        elif self.config['env_config']['opt_method'] == 'sgd':
            self.num_actions = 1
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9).minimize(loss)

    sess = tf.Session()
    loader = tf.train.Saver()
    if os.path.exists(save_file):
        loader.restore(sess, save_file)
    else:
        sess.run(tf.initialize_all_variables())
    for i in xrange(max_iter):
        X = np.random.uniform(-max_x, max_x, 
                (bsize, actor_config['num_features'], num_vars, dim_features))
        feed = {inputs: X}
        _, loss_v = sess.run([train_op, loss], feed)
        if i % 10 == 0:
            print "iter %d, loss %f" % (i, loss_v)

    '''
    # create variables for actor_target
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    new_vars = []
    with tf.variable_scope('actor_target'):
        for v in variables:
            vname = v.name.replace('actor/', '').split(':')[0]
            new_vars.append(tf.get_variable(vname, initializer=v))
    sess.run(tf.initialize_variables(new_vars))
    '''

    saver = tf.train.Saver()
    saver.save(sess, save_file)
    print "save to " + save_file


if __name__ == "__main__":
    main()

