import tensorflow as tf
import numpy as np
import gym
from estimators import Estimator

def init_world(name):
    worlds = {
            'gym_world': GymWorld,
            'Pong-v0': PongWorld,
            'function': FunctionWorld
            }
    if name not in worlds:
        return worlds['gym_world'](name)
    else:
        return worlds[name](name)

class World(object):
    def __init__(self, name):
        self.name = name

    def eps_end(self):
        return None

class GymWorld(World):
    def __init__(self, name):
        super(GymWorld, self).__init__(name)
        self.env = gym.make(name)
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
    def __init__(self, name):
        super(PongWorld, self).__init__(name)

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
    def __init__(self):
        # build model to be optimized
        pass

    def gen_feed(self):
        pass

    def build_state(self):
        self.state = tf.concat(0, 
                [tf.expand_dims(tf.concat(0, self.variables), 0), 
                tf.expand_dims(tf.concat(0, self.grads), 0)])

    def build_train(self):
        self.grads = tf.gradients(self.loss, self.variables)
        self.grad_scale = tf.reduce_sum([tf.nn.l2_loss(g) for g in self.grads])
        self.num_vars = len(self.variables)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=1.0,
                decay=0.9)
        self.t_action = tf.placeholder(dtype=tf.float32, shape=self.num_vars, name='train')
        self.train_op = optimizer.apply_gradients([(self.t_action[i] * g, v) for i, (v, g) in enumerate(zip(self.variables, self.grads))])

class SimpleFunction(Function):
    def __init__(self):
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

    def gen_feed(self):
        return {}

class SimpleNNFunction(Function):
    def __init__(self):
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
        self.variables = tf.trainable_variables()
        self.build_train()
        self.build_state()

    def build_state(self):
        variables = []
        grads = []
        for v, g in zip(self.variables, self.grads):
            variables.append(tf.reshape(v, (-1,)))
            grads.append(tf.reshape(g, (-1,)))
        self.state = tf.concat(0, 
                [tf.expand_dims(tf.concat(0, variables), 0), 
                tf.expand_dims(tf.concat(0, grads), 0)])

    def gen_feed(self):
        return {self.t_X: self.X, self.t_y: self.y}

    def gen_data(self):
        size = 400
        ydim = 5
        Xdim = 2
        X_base = np.random.rand(ydim, Xdim) * 20
        y = np.random.randint(0, ydim, size).astype('int32')
        X = np.zeros((size, Xdim)).astype('float32')
        for i in range(ydim):
            X[y == i] = X_base[i]
        X += np.random.randn(size, Xdim)
        return X, y, ydim

class FunctionWorld(World):
    def __init__(self, name):
        # experiment to learn sgd
        super(FunctionWorld, self).__init__(name)
        self.stop_thres = 1e-1
        #self.Func = SimpleFunction()
        self.Func = SimpleNNFunction()
        self.variables = self.Func.variables
        self.loss = self.Func.loss
        self.train_op = self.Func.train_op
        self.t_action = self.Func.t_action
        self.state = self.Func.state
        self.num_vars = self.Func.num_vars
        self.grad_scale = self.Func.grad_scale

        self.action_low, self.action_high  = np.array([0] * self.num_vars), np.array([1] * self.num_vars)
        self.action_space = ContSpace(self.action_low.shape, self.action_low, self.action_high)
        self.action_dim = self.action_space.shape
        self.observation_space = ContSpace(self.state.get_shape().as_list())
        self.observation_dims = self.observation_space.shape

        
        self.sess = tf.Session()

    def reset(self):
        self.step_counter = 0
        self.sess.run(tf.initialize_all_variables())
        feed = self.Func.gen_feed()
        state, value = self.sess.run([self.state, self.loss], feed_dict=feed)
        self.last_value = value
        self.init_value = value
        return state

    def eps_end(self):
        def ee(done, reward, state=None):
            return np.sum(state[1] ** 2) < self.stop_thres
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
        done = grad_scale < self.stop_thres
        #reward = inc / grad_scale if not done else inc / grad_scale - self.step_counter
        #reward = inc if not done else inc - self.step_counter
        reward = inc if not done else inc + self.init_value - value
        self.step_counter += 1
        #reward = -1 #TODO
        return state, reward, done, None
