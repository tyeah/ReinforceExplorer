import tensorflow as tf
import numpy as np
import gym
from estimators import Estimator
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import slim

def init_world(name, **kwargs):
    worlds = {
            'gym_world': GymWorld,
            'Pong-v0': PongWorld,
            'function': FunctionWorld
            }
    if name not in worlds:
        return worlds['gym_world'](name, **kwargs)
    else:
        return worlds[name](name, **kwargs)

class World(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def eps_end(self):
        return None

class GymWorld(World):
    def __init__(self, name):
        super(GymWorld, self).__init__(name, **kwargs)
        self.env = gym.make(name, **kwargs)
        self.action_space = self.env.action_space
        self.action_dim = self.env.action_space.n
        self.step = self.env.step
        self.reset = self.env.reset
        self.render = self.env.render
        self.observation_dims = self.env.observation_space.shape

    def eps_end(self):
        def ee(done, reward, state=None):
            return done
        return ee

class PongWorld(GymWorld):
    def __init__(self, name, **kwargs):
        super(PongWorld, self).__init__(name, **kwargs)

    def eps_end(self):
        def ee(done, reward, state=None):
            return reward != 0
        return ee

    def preproc(self, I):
        I = I[35:195][::2, ::2]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(np.float).ravel()

    def reset():
        state = self.env.reset()
        return self.preproc(state)

    def step(self, action):
        state, reward, done, z = env.step(action)
        return self.preproc(state), reward, done, z

class ContSpace(object):
    def __init__(self, shape, low=None, high=None):
        self.low = np.zeros(shape) if low is None else np.array(low)
        self.high = np.zeros(shape) if high is None else np.array(high)
        self.shape = tuple(shape)

class Function(object):
    # Function to be optimized
    def __init__(self, **kwargs):
        # build model to be optimized
        self.kwargs = kwargs

    def gen_feed(self):
        pass

    def dist_grads(self, grads_flatten):
        return [(tf.reshape(grads_flatten[self.dims_bins[i]: self.dims_bins[i+1]], 
            self.dims[i]), v)  
            for i, v in enumerate(self.variables)]

    def dist_lr(self, grads_flatten):
        return [(tf.reshape(grads_flatten[self.dims_bins[i]: self.dims_bins[i+1]], 
            self.dims[i]) * self.grads[i], v)  
            for i, v in enumerate(self.variables)]

    def build_train(self):
        self.grads = tf.gradients(self.loss, self.variables)
        #self.grad_scale = tf.reduce_sum([tf.nn.l2_loss(g) for g in self.grads])
        self.grad_scale = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in self.grads])
        self.num_vars = len(self.variables)
        self.dims = [v.get_shape().as_list() for v in self.variables]
        self.dims_bins = np.cumsum([0] + [np.prod(d) for d in self.dims])
        if self.kwargs['action'] == 'learning_rate':
            self.num_actions = self.num_vars
        elif self.kwargs['action'] in ['step', 'coordinate_lr'] :
            self.num_actions = sum([np.prod(d) for d in self.dims])
        elif self.kwargs['action'] == 'params':
            if self.kwargs['opt_method'] == 'rmsprop':
                self.num_actions = 4
            elif self.kwargs['opt_method'] == 'sgd':
                self.num_actions = 1
        self.action_low, self.action_high  = np.array([0] * self.num_vars), np.array([1] * self.num_vars)

        self.t_action = tf.placeholder(dtype=tf.float32, shape=self.num_actions, name='train')
        if self.kwargs['action'] != 'params':
            #optimizer = tf.train.RMSPropOptimizer(learning_rate=1.0,
            #        decay=0.9)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        else:
            if self.kwargs['opt_method'] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.t_action[0],
                        decay=self.t_action[1], momentum=self.t_action[2],
                        epsilon=self.t_action[3])
            elif self.kwargs['opt_method'] == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.t_action[0])
        if self.kwargs['action'] == 'learning_rate':
            self.train_op = optimizer.apply_gradients([(self.t_action[i] * g, v) for i, (v, g) in enumerate(zip(self.variables, self.grads))])
        elif self.kwargs['action'] == 'step':
            self.train_op = optimizer.apply_gradients(self.dist_grads(self.t_action))
        elif self.kwargs['action'] == 'coordinate_lr':
            self.train_op = optimizer.apply_gradients(self.dist_grads(self.t_action))
        elif self.kwargs['action'] == 'params':
            self.train_op = optimizer.apply_gradients([(g, v) for (v, g) in zip(self.variables, self.grads)])

    def build_state(self):
        variables = []
        grads = []
        for v, g in zip(self.variables, self.grads):
            variables.append(tf.reshape(v, (-1,)))
            grads.append(tf.reshape(g, (-1,)))
        if self.kwargs['state'] == 'gradient':
            self.state = tf.expand_dims(tf.concat(0, grads), 0)
        elif self.kwargs['state'] == 'variable_gradient':
            self.state = tf.concat(0, 
                    [tf.expand_dims(tf.concat(0, variables), 0), 
                    tf.expand_dims(tf.concat(0, grads), 0)])
        elif self.kwargs['state'] == 'loss':
            self.state = tf.reshape(self.loss, (1,))

class SimpleFunction(Function):
    def __init__(self, **kwargs):
        super(SimpleFunction, self).__init__(**kwargs)
        with tf.variable_scope('Variables'):
            self.w = tf.get_variable('w', 
                    (10,), tf.float32, tf.random_normal_initializer(mean=1., stddev=1))
            self.w1 = tf.get_variable('w1', 
                    (10,), tf.float32, tf.random_normal_initializer(mean=1., stddev=1))
        self.variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="Variables")
        self.loss = tf.reduce_sum([tf.reduce_sum(tf.sin(v) + tf.square(v)) for v in self.variables])
        #self.loss = tf.reduce_sum(self.w)
        self.build_train()
        self.build_state()

class QuadraticFunction(Function):
    def __init__(self, **kwargs):
        super(QuadraticFunction, self).__init__(**kwargs)
        num_vars = 20
        A0 = np.random.randn(num_vars, num_vars).astype('float32')
        A = A0.T.dot(A0)
        A += np.eye(num_vars) * 0.1
        b = np.random.randn(num_vars).astype('float32')
        #A = np.eye(num_vars).astype('float32')
        #b = np.ones(num_vars).astype('float32')
        with tf.variable_scope('Variables'):
            self.x = tf.get_variable('x', 
                    (num_vars,), tf.float32, tf.random_normal_initializer(mean=1., stddev=1))
        self.loss = 0.5 * tf.squeeze(tf.matmul(tf.reshape(self.x, (1, -1)), 
            tf.matmul(A, tf.reshape(self.x, (-1, 1))))) + tf.reduce_sum(b * self.x)
        self.variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="World")
        self.build_train()
        self.build_state()

    def gen_feed(self):
        return {}

class SimpleNNFunction(Function):
    def __init__(self, **kwargs):
        super(SimpleNNFunction, self).__init__(**kwargs)
        self.X, self.y, ydim = self.gen_data()
        self.t_X = tf.placeholder(dtype=tf.float32, shape=(None, self.X.shape[1]), name='X')
        self.t_y = tf.placeholder(dtype=tf.int32, shape=(None,), name='y')
        estimator_params = {
            "name": "fc", 
            "num_hids": [20],
            "trainable": True
         }
        self.logits = Estimator(estimator_params['name']).get_estimator(
                inputs=[self.t_X], num_out=ydim, 
                scope='estimator', 
                **estimator_params)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits , self.t_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='World')
        self.build_train()
        self.build_state()

    def gen_feed(self):
        return {self.t_X: self.X, self.t_y: self.y}

    def gen_data(self):
        size = 40
        ydim = 5
        Xdim = 2
        X_base = np.random.rand(ydim, Xdim) * 20
        y = np.random.randint(0, ydim, size).astype('int32')
        X = np.zeros((size, Xdim)).astype('float32')
        for i in range(ydim):
            X[y == i] = X_base[i]
        X += np.random.randn(size, Xdim)
        return X, y, ydim

class MNISTFunction(Function):
    def __init__(self, **kwargs):
        super(MNISTFunction, self).__init__(**kwargs)
        self.batch_size = kwargs['batch_size']

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.gen_train = self.mnist.train
        X, y = self.gen_train.next_batch(1)
        self.x_dim = X.shape[1]

        self.t_X = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name='MNIST_X')
        self.t_y = tf.placeholder(dtype=tf.int32, shape=(None,), name='MNIST_y')
        self.logits = Estimator('fc').get_estimator(
                inputs=[self.t_X], num_out=10, 
                num_hids=[500, 300],
                trainable=True,
                scope='mnist_lenet')
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits , self.t_y))
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='World')
        self.build_train()
        self.build_state()

    def gen_feed(self):
        X, y = self.gen_train.next_batch(self.batch_size)
        return {self.t_X: X, self.t_y: y}

class FunctionWorld(World):
    def __init__(self, name, **kwargs):
        # experiment to learn sgd
        super(FunctionWorld, self).__init__(name, **kwargs)
        self.base_stop_thres = kwargs['stop_grad'] #simplenn
        self.stop_thres = self.base_stop_thres
        #self.stop_thres = 1e0 #quad
        #self.Func = SimpleFunction()
        with tf.variable_scope('World'):
            if 'func' not in kwargs or kwargs['func'] == 'simplenn':
                self.Func = SimpleNNFunction(**kwargs)
            elif 'func' not in kwargs or kwargs['func'] == 'quad':
                self.Func = QuadraticFunction(**kwargs)
            elif 'func' not in kwargs or kwargs['func'] == 'mnist':
                self.Func = MNISTFunction(**kwargs)
        self.variables = self.Func.variables
        self.loss = self.Func.loss
        self.train_op = self.Func.train_op
        self.t_action = self.Func.t_action
        self.state = self.Func.state
        self.num_vars = self.Func.num_vars
        self.grad_scale = self.Func.grad_scale
        self.num_actions = self.Func.num_actions

        self.action_low, self.action_high  = np.array([0] * self.num_actions), np.array([1] * self.num_actions)
        self.action_space = ContSpace(self.action_low.shape, self.action_low, self.action_high)
        self.action_dim = self.action_space.shape
        self.observation_space = ContSpace(self.state.get_shape().as_list())
        self.observation_dims = self.observation_space.shape

        self.episode_counter = 0

        
        self.sess = tf.Session()

    def reset(self):
        self.episode_counter += 1
        if self.stop_thres > self.base_stop_thres * 0.01:
            self.stop_thres *= 0.999
        self.step_counter = 0
        #tf.reset_default_graph() #??????????????????????
        self.sess.run(tf.initialize_all_variables())
        feed = self.Func.gen_feed()
        state, value = self.sess.run([self.state, self.loss], feed_dict=feed)
        self.last_value = value
        self.init_value = value
        return state

    def eps_end(self):
        if self.kwargs['state'] in ['gradient', 'state', 'variable_gradient']:
            def ee(done, reward, state=None):
                return np.sum(state[-1] ** 2) < self.stop_thres
        else:
            def ee(done, reward, state=None):
                return done
        return ee

    def step(self, action):
        feed = self.Func.gen_feed()
        feed.update({self.t_action: action})
        #feed.update({self.t_action: np.full(self.num_vars, 0.01)})
        self.sess.run(self.train_op, feed_dict=feed)
        #self.sess.run(self.train_op, feed_dict={self.t_action: np.full(self.num_vars, 0.1)})
        state, value, grad_scale = self.sess.run([self.state, self.loss, self.grad_scale], feed_dict=feed)
        #print value, grad_scale

        inc = (self.last_value - value)
        self.last_value = value
        #print "grad_scale: %f, value: %f" % (grad_scale, value)
        info = True
        if self.kwargs['func'] != 'mnist':
            if grad_scale < self.stop_thres:
                done = True
                reward = inc + self.init_value - value
                #reward = inc
                print value, reward, grad_scale, self.stop_thres
            elif value - self.init_value > 500:
                done = True
                reward = - 1000
                #done = False
                #reward = inc
                info = False # abnormal exit
            else:
                done = False
                reward = inc
        else:
            if self.step_counter >= self.kwargs['max_iter']:
                done = True
                reward = inc
            elif value - self.init_value > 50:
                done = True
                reward = - 1000
                info = False # abnormal exit
            else:
                done = False
                reward = inc
            if self.step_counter % 100 == 0:
                print value, reward, grad_scale, self.step_counter, action
        '''
        done = grad_scale < self.stop_thres
        #reward = inc / grad_scale if not done else inc / grad_scale - self.step_counter
        #reward = inc if not done else inc - self.step_counter
        reward = inc if not done else inc + self.init_value - value
        '''
        self.step_counter += 1
        #reward = -1 #TODO
        return state, reward, done, info
