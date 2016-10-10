import tensorflow as tf
import numpy as np
from inner_states import InnerState
from estimators import Estimator

class Agent(object):
    '''
    need to keep track on the experiences and decide when to update
    inner state should be separated with the outer state
    e.g., inner state could be the hidden state of an RNN, and in this case
    the outer state is the input to the inner state
    '''
    def __init__(self, observation_dims, action_dim, eps_end=None):
        '''
        eps_end should be a function of reward (and state), which is 
        specific to each environment, returning True if
        the episode should end (useful in monte carlo methods). e.g., in
        pong or go, reward != 1 means the end of the episode. in TD 
        method this is not necessary. 
        eps_end should be a function of state and reward, which is 
        '''
        self.observation_dims = observation_dims
        self.action_dim = action_dim
        self.eps_end = eps_end

    def init_state(self, state):
        self.inner_state = InnerState(state)

    def action(self):
        # tack action based on current state
        return np.random.randint(0, self.action_dim)

    def experience(self, state, reward):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        print self.eps_end(reward, state)

        # if cond;
        #     self.update
        return 0

    def update(self):
        # learn from experiences
        return 0

    def build_train(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                decay=0.9, epsilon=1e-5)
        tvars = tf.trainable_variables()
        if self.clip_norm <= 0:
            grads = tf.gradients(self.loss, tvars)
        else:
            grads, norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_norm)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

class PGAgent(Agent):
    '''
    Policy Gradient
    '''
    def __init__(self, observation_dims, action_dim, eps_end=None):
        super(PGAgent, self).__init__(observation_dims, action_dim, eps_end)
        self.discount_rate = 0.99
        #max_gradient=5,
        self.reg_param = 1e-2
        self.init_learning_rate = 1e-1
        self.init_exp_rate = 0.5
        self.anneal_step_exp = 1000
        self.anneal_step_lr = 1000
        self.anneal_base_exp = 0.5
        self.anneal_base_lr = 0.5
        self.min_lr = 1e-5
        self.clip_norm = -1
        self.store_eps = 2
        self.store_size = 100

        if self.store_eps is not None:
            self.cond = lambda: np.sum(self.rollouts['eps_end_masks']) >= self.store_eps
        elif self.store_size is not None:
            self.cond = lambda: len(self.rollouts['rewards']) >= self.store_size
        self.reset_buffer()
        self.sess = tf.Session()
        self.build_model()
        self.sess.run(tf.initialize_all_variables())

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od) for od in self.observation_dims]
        self.t_action = tf.placeholder(dtype=tf.int32, shape=(None,)) # for discrete action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.exp_rate = self.init_exp_rate * (self.anneal_base_exp ** 
                tf.cast(tf.floordiv(self.global_step, self.anneal_step_exp), tf.float32))
        self.learning_rate = tf.maximum(self.init_learning_rate * (self.anneal_base_lr ** 
                tf.cast(tf.floordiv(self.global_step, self.anneal_step_lr), tf.float32)), self.min_lr)

        self.action_scores = Estimator('cnn').get_estimator(
                inputs=self.t_state, num_out=self.action_dim, 
                num_layers=3,
                reuse=False, trainable=True, scope='policy_network')
        policy_network_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
        self.action_probs = tf.nn.softmax(self.action_scores)
        self.explore = tf.less(tf.random_uniform([batch_size]), self.exp_rate)
        self.action_sampler = tf.select(self.explore, 
                tf.multinomial(random_action_probs, num_samples=1),
                tf.multinomial(self.action_probs, num_samples=1))
        # TODO: seed?

        self.reinforce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.action_scores, self.t_action) * self.t_discounted_reward)

        self.reg_loss = tf.reduce_mean([tf.nn.l2_loss(v) for v in policy_network_variables])
        self.loss = self.reinforce_loss + self.reg_param * self.reg_loss
        self.build_train()

    def reset_buffer(self):
        self.rollouts = {
                'states': [[] for _ in xrange(len(self.observation_dims))],
                'rewards': [],
                'actions': [],
                'eps_end_masks': []
                }

    def update_buffer(self, state, reward, action, eps_end_mask):
        for s in self.rollouts['states']:
            s.append(state)
        self.rollouts['rewards'].append(reward)
        self.rollouts['actions'].append(action)
        self.rollouts['eps_end_masks'].append(eps_end_mask)

    def init_state(self, state):
        self.inner_state = InnerState(state)

    def action(self):
        # tack action based on current state
        action = self.sess.run(self.action_sampler, feed_dict=dict([(s_t, [s]) for s_t, s in zip(self.t_state, self.inner_state.current_state)]))[0][0]
        self.rollouts['actions'].append(action)
        return action

    def experience(self, state, reward):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        for s in self.rollouts['states']:
            s.append(state)
        self.rollouts['rewards'].append(reward)
        self.rollouts['eps_end_masks'].append(self.eps_end(reward, state))
        # TODO: ansemble states, discounti rewards, actions, do update

        if self.cond():
            self.update()
            self.reset_buffer()
        return 0

    def update(self):
        # learn from experiences
        discounted_reward = np.zeros_like(self.rollouts['rewards'])
        for i in reversed(xrange(0, len(self.rollouts['rewards']))):
            if self.rollouts['eps_end_masks'][i]:
                dr = 1
            else:
                dr *= self.discount_rate
            discounted_reward[i] = dr
        feed = dict([(s_t, s) for s_t, s in zip(self.t_state, self.rollouts['states'])] +
                [(self.t_discounted_reward, discounted_reward), (self.t_action, self.rollouts['actions'])])
        _ = self.sess.run(self.train_op, feed_dict=feed)
        reward_array = np.array(self.rollouts['rewards'])
        print('avg reward: %f' % np.mean(reward_array[reward_array != 0]))
