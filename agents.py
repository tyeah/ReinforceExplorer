import tensorflow as tf
import numpy as np
from inner_states import InnerState
from estimators import Estimator
from collections import deque
import os


def init_agent(agent_name):
    agents = {
            'naive': Agent,
            'policy_gradient': PGAgent
            }
    return agents[agent_name]

class Agent(object):
    '''
    need to keep track on the experiences and decide when to update
    inner state should be separated with the outer state
    e.g., inner state could be the hidden state of an RNN, and in this case
    the outer state is the input to the inner state
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
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
        self.learning = learning
        self.config = config

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
        if self.config["clip_norm"] <= 0:
            grads = tf.gradients(self.loss, tvars)
        else:
            grads, norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config["clip_norm"])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

class PGAgent(Agent):
    '''
    Policy Gradient
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
        super(PGAgent, self).__init__(observation_dims, action_dim, config, eps_end, learning)
        '''
        self.config["discount_rate"] = 0.99
        self.config["reg_param"] = 1e-2
        self.config["init_learning_rate"] = 1e-2
        self.config["init_exp_rate"] = 0.5
        self.config["anneal_step_exp"] = 1000
        self.config["anneal_step_lr"] = 10000
        self.config["anneal_base_exp"] = 0.5
        self.config["anneal_base_lr"] = 0.5
        self.config["min_lr"] = 1e-5
        self.config["clip_norm"] = 5
        self.config["store_eps"] = 4
        self.config["store_size"] = 100
        self.config["batch_size"] = 100
        self.config["save_step"] = 10000
        self.config["save_dir"] = 'weights/pg'
        #weights = os.path.abspath('.') + '/pg_save.ckpt'
        weights = None
        '''
        
        self.eps_counter = 0
        self.save_file = os.path.join(self.config["save_dir"], 'save.ckpt')

        self.reward_queue = deque(maxlen=100)
        if self.config["store_eps"] is not None:
            self.cond = lambda: np.sum(self.rollouts['eps_end_masks']) >= self.config["store_eps"]
        elif self.config["store_size"] is not None:
            self.cond = lambda: len(self.rollouts['rewards']) >= self.config["store_size"]
        self.reset_buffer()
        self.sess = tf.Session()
        self.build_model()
        self.saver = tf.train.Saver()
        if self.config["weights"] is not None:
            self.saver.restore(self.sess, self.config["weights"])
        else:
            self.sess.run(tf.initialize_all_variables())

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od) for od in self.observation_dims]
        self.t_action = tf.placeholder(dtype=tf.int32, shape=(None,)) # for discrete action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.exp_rate = self.config["init_exp_rate"] * (self.config["anneal_base_exp"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_exp"]), tf.float32))
        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), tf.float32)), self.config["min_lr"])

        self.action_scores = Estimator('cnn').get_estimator(
                inputs=self.t_state, num_out=self.action_dim, 
                num_cnn_layers=5, num_fc_layers=2,
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
        self.loss = self.reinforce_loss + self.config["reg_param"] * self.reg_loss
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
            self.eps_counter += np.sum(self.rollouts['eps_end_masks'])
            if self.learning:
                self.update()
            self.reset_buffer()
        return 0

    def gen_feed(self, discounted_reward):
        num_batches = int(len(self.rollouts['rewards']) // self.config["batch_size"])
        res = (len(self.rollouts['rewards']) % self.config["batch_size"]) > 0
        for i in xrange( num_batches + res):
            feed = dict([(s_t, s[i:(i+self.config["batch_size"])]) for s_t, s in zip(self.t_state, self.rollouts['states'])] +
                    [(self.t_discounted_reward, discounted_reward[i:(i+self.config["batch_size"])]), (self.t_action, self.rollouts['actions'][i:(i+self.config["batch_size"])])])
            yield feed


    def update(self):
        # learn from experiences
        discounted_reward = np.zeros_like(self.rollouts['rewards'])
        for i in reversed(xrange(0, len(self.rollouts['rewards']))):
            if self.rollouts['eps_end_masks'][i]:
                dr = 0
            dr = dr * self.config["discount_rate"] + self.rollouts['rewards'][i]
            discounted_reward[i] = dr
        self.reward_queue.append(np.mean(discounted_reward))
        feeds = self.gen_feed(discounted_reward)
        for feed in feeds: 
            _ = self.sess.run([self.global_step, self.train_op], feed_dict=feed)
        reward_array = np.array(self.rollouts['rewards'])
        #print('avg reward: %f' % np.mean(reward_array[reward_array != 0]))
        print('after episode %d, avg reward: %10.7f, accumulated avg reward: %f, successes: %d' % (self.eps_counter, np.mean(discounted_reward), np.mean(self.reward_queue), np.sum(reward_array > 0)))
        if self.eps_counter % self.config["save_step"]:
            self.saver.save(self.sess, self.save_file)
