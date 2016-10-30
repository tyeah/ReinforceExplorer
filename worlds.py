import tensorflow as tf
import numpy as np
import gym

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

class FunctionWorld(World):
    def __init__(self, name):
        # experiment to learn sgd
        super(FunctionWorld, self).__init__(name)
        self.stop_thres = 1e-1
        with tf.variable_scope('Variables'):
            self.w = tf.get_variable('w', 
                    (10,), tf.float32, tf.random_normal_initializer(mean=1., stddev=1))
            self.w1 = tf.get_variable('w1', 
                    (10,), tf.float32, tf.random_normal_initializer(mean=1., stddev=1))
        self.variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope="Variables")
        self.loss = tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in self.variables])
        #self.loss = tf.reduce_sum(self.w)

        self.grads = tf.gradients(self.loss, self.variables)
        self.grad_scale = tf.reduce_sum([tf.nn.l2_loss(g) for g in self.grads])
        self.num_vars = len(self.variables)
        self.state = tf.concat(0, 
                [tf.expand_dims(tf.concat(0, self.variables), 0), 
                tf.expand_dims(tf.concat(0, self.grads), 0)])

        self.action_low, self.action_high  = np.array([0] * self.num_vars), np.array([1] * self.num_vars)
        self.action_space = ContSpace(self.action_low.shape, self.action_low, self.action_high)
        self.action_dim = self.action_space.shape
        self.observation_space = ContSpace(self.state.get_shape().as_list())
        self.observation_dims = self.observation_space.shape

        optimizer = tf.train.RMSPropOptimizer(learning_rate=1.0,
                decay=0.9)
        self.t_action = tf.placeholder(dtype=tf.float32, shape=self.num_vars)
        #self.train_op = optimizer.apply_gradients([(self.t_action[i] * g, v) for i, (v, g) in enumerate(zip(self.variables, self.grads))])
        self.train_op = optimizer.apply_gradients([(self.t_action[i] * g, v) for i, (v, g) in enumerate(zip(self.variables, self.grads))])
        
        self.sess = tf.Session()

    def reset(self):
        self.step_counter = 0
        self.sess.run(tf.initialize_all_variables())
        state, value = self.sess.run([self.state, self.loss])
        self.last_value = value
        self.init_value = value
        return state

    def eps_end(self):
        def ee(done, reward, state=None):
            return np.sum(state[1] ** 2) < self.stop_thres
        return ee

    def step(self, action):
        self.sess.run(self.train_op, feed_dict={self.t_action: action})
        #self.sess.run(self.train_op, feed_dict={self.t_action: np.full(self.num_vars, 50)})
        state, value, grad_scale = self.sess.run([self.state, self.loss, self.grad_scale])
        #print value, grad_scale

        inc = (self.last_value - value)
        self.last_value = value
        done = grad_scale < self.stop_thres
        #reward = inc / grad_scale if not done else inc / grad_scale - self.step_counter
        #reward = inc if not done else inc - self.step_counter
        reward = inc if not done else inc + self.init_value - value
        self.step_counter += 1
        #reward = -1 #TODO
        return state, reward, done, None
