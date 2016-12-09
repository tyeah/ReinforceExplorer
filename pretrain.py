import tensorflow as tf
import numpy as np
from estimators import Estimator
import json

def main():
    config_file = "configs/ddpgcont_parallel_func.json"
    #max_iter = 10000
    max_iter = 100
    bsize, num_vars = 40, 200
    max_x = 10
    save_file = 'weights/' + config_file.split('/')[-1].split('.')[0] + '/pretrain.ckpt'


    config = json.load(open(config_file))

    actor_config = config['estimator_params']['policy_network']
    inputs = tf.placeholder(dtype=tf.float32, shape=(bsize, 
        actor_config['num_features'], num_vars, 1))
    actor = Estimator(config['estimator_params']
            ['policy_network']['name']).get_estimator(
            inputs=[inputs],
            scope='actor',
            **actor_config)
    loss = tf.reduce_sum(tf.square(tf.squeeze(actor) - inputs[:, -1, :, 0]))
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(loss)

    sess = tf.Session()
    loader = tf.train.Saver()
    #sess.run(tf.initialize_all_variables())
    loader.restore(sess, save_file)
    for i in xrange(max_iter):
        X = np.random.uniform(-max_x, max_x, 
                (bsize, actor_config['num_features'], num_vars, 1))
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


if __name__ == "__main__":
    main()

