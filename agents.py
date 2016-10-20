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
        self.summary_scalars = {}
        self.summary_histograms = {}

    def init_state(self, state):
        self.inner_state = InnerState(state)

    def action(self):
        # tack action based on current state
        return np.random.randint(0, self.action_dim)

    def experience(self, state, reward, done):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        print self.eps_end(done, reward, state)

        # if cond;
        #     self.update
        return 0

    def update(self):
        # learn from experiences
        return 0

    def build_train(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                decay=0.9)
        tvars = tf.trainable_variables()
        if self.config["clip_norm"] <= 0:
            grads = tf.gradients(self.loss, tvars)
        else:
            grads, norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config["clip_norm"])
        gradients = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    def add_summary(self):
        for k, v in self.summary_scalars.iteritems():
            tf.scalar_summary(k, v)
        for k, v in self.summary_histograms.iteritems():
            tf.histogram_summary(k, v)
        self.summary_merged =tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.config['save_dir'] + '/summary', self.loss.graph)
        num_params = 0
        for v in tf.trainable_variables():
            #print v.name, v.get_shape()
            num_params += np.prod(v.get_shape().as_list())
        print 'total number of params: %d' % num_params

    def add_reg(self):
        self.reg_loss = {}
        if 'l2' in self.config and self.config['l2'] > 0:
            self.reg_loss['l2'] = self.config['l2'] * tf.reduce_mean([tf.nn.l2_loss(v) for v in policy_network_variables])
        if 'l1' in self.config and self.config['l1'] > 0:
            self.reg_loss['l1'] = self.config['l1'] * tf.reduce_mean([tf.nn.l1_loss(v) for v in policy_network_variables])

class PGAgent(Agent):
    '''
    Policy Gradient
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
        super(PGAgent, self).__init__(observation_dims, action_dim, config, eps_end, learning)

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
        '''
        self.add_summary()
        '''

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od) for od in self.observation_dims]
        self.t_action = tf.placeholder(dtype=tf.int32, shape=(None,)) # for discrete action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        '''
        self.exp_rate = self.config["init_exp_rate"] * (self.config["anneal_base_exp"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_exp"]), tf.float32))
        '''

        #tf.cast(tf.maximum(self.config['anneal_step_exp'] - self.global_step, 0), tf.float32)
        self.exp_rate = tf.cast(tf.maximum(self.config['anneal_step_exp'] - self.global_step, 0), tf.float32) / (1.0 * self.config['anneal_step_exp']) * (self.config["init_exp_rate"] - self.config["min_exp"]) + self.config["min_exp"]

        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), tf.float32)), self.config["min_lr"])

        self.action_scores = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state, num_out=self.action_dim, 
                scope='policy_network', 
                **self.config['estimator_params']['policy_network'])
        policy_network_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
        self.action_probs = tf.nn.softmax(self.action_scores)
        self.explore = tf.less(tf.random_uniform([batch_size]), self.exp_rate)
        self.action_sampler = tf.select(self.explore, 
                tf.multinomial(random_action_probs, num_samples=1),
                tf.multinomial(self.action_probs, num_samples=1))
        # TODO: seed?
        # TODO: how to measure global_step?

        dr_mean, dr_var = tf.nn.moments(self.t_discounted_reward, axes=[0])
        dr_std = tf.sqrt(dr_var)
        t_discounted_reward_reparamed = self.t_discounted_reward
        #t_discounted_reward_reparamed = (self.t_discounted_reward - dr_mean) / dr_std
        #self.dr_mean, self.dr_std, self.t_discounted_reward_reparamed = dr_mean, dr_std, t_discounted_reward_reparamed
        # reparameterization trick
        self.reinforce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.action_scores , self.t_action) * t_discounted_reward_reparamed)
        self.add_reg()
        self.loss = tf.identity(self.reinforce_loss)
        for v in self.reg_loss.values():
            self.loss += v
        self.build_train()
        '''
        ####################################
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                decay=0.9)
        self.reinforce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.action_scores , self.t_action))
        self.loss = self.reinforce_loss
        self.gradients = optimizer.compute_gradients(self.loss)
        for i, (grad, var) in enumerate(self.gradients):
            if grad is not None:
                #print var.name, var.get_shape(), grad.get_shape(), (grad * self.discounted_rewards).get_shape()
                self.gradients[i] = (grad * t_discounted_reward_reparamed, var)
        self.train_op = optimizer.apply_gradients(self.gradients, global_step=self.global_step)
        ####################################
        '''

        '''
        self.summary_scalars['reinforce_loss'] = self.reinforce_loss
        self.summary_scalars['loss'] = self.loss
        for k, v in self.reg_loss.iteritems():
            self.summary_scalars[k] = v
        self.summary_scalars['discounted_reward'] = tf.reduce_mean(self.t_discounted_reward)
        self.summary_scalars['learning_rate'] = self.learning_rate
        self.summary_scalars['exp_rate'] = self.exp_rate
        for v in policy_network_variables:
            self.summary_histograms[v.name] = v
        '''


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
        self.inner_state = InnerState(state, **self.config['inner_state_params'])

    def action(self):
        # tack action based on current state
        action = self.sess.run(self.action_sampler, feed_dict=dict([(s_t, [s]) for s_t, s in zip(self.t_state, self.inner_state.current_state)]))[0][0]
        self.rollouts['actions'].append(action)
        return action

    def experience(self, state, reward, done):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        # TODO: change the stored states as inner states
        current_inner_state = self.inner_state.get_current_state()
        for sidx, s in enumerate(self.rollouts['states']):
            s.append(current_inner_state[sidx])
        self.rollouts['rewards'].append(reward)
        self.rollouts['eps_end_masks'].append(self.eps_end(done, reward, state))
        # TODO: ansemble states, discounti rewards, actions, do update

        if self.cond():
            self.eps_counter += np.sum(self.rollouts['eps_end_masks'])
            if self.learning:
                self.update()
            self.reset_buffer()
        return 0

    def gen_feed(self, discounted_reward):
        '''
        for v in self.rollouts.values():
            v = v[:-1]
        '''
        num_batches = int(np.ceil(len(self.rollouts['rewards']) * 1.0 / self.config["batch_size"]))
        for i in xrange(num_batches):
            feed = dict([(s_t, np.array(s[i:(i+self.config["batch_size"])])) for s_t, s in zip(self.t_state, self.rollouts['states'])] +
                    [(self.t_discounted_reward, np.array(discounted_reward[i:(i+self.config["batch_size"])])), (self.t_action, np.array(self.rollouts['actions'][i:(i+self.config["batch_size"])]))])
            yield feed


    def update(self):
        # learn from experiences
        discounted_reward = np.zeros_like(self.rollouts['rewards'])
        for i in reversed(xrange(0, len(self.rollouts['rewards']))):
            if self.rollouts['eps_end_masks'][i]:
                dr = 0
            dr = dr * self.config["discount_rate"] + self.rollouts['rewards'][i]
            discounted_reward[i] = dr
            
        # TODO: when to update, how to set batch, non-iid sgd, loss func
        self.reward_queue.append(np.mean(discounted_reward))
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)
        #print discounted_reward

        feeds = self.gen_feed(discounted_reward)
        '''
        if self.eps_counter % self.config["log_step"] == 0:
            for i, feed in enumerate(feeds): 
                if i == 0:
                    summary, _ = self.sess.run([self.summary_merged, self.train_op], feed_dict=feed)
                    self.writer.add_summary(summary, self.eps_counter)
                else:
                    _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        else:
            for i, feed in enumerate(feeds): 
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        '''
        for feed in feeds:
            #print([v.shape for v in feed.values()])
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        #print("global step: %d, exp rate: %f" % tuple(self.sess.run([self.global_step, self.exp_rate])))
        #print("global step: %d, exp rate: %f" % (self.eps_counter, self.sess.run(self.exp_rate)))
        #reward_array = np.array(self.rollouts['rewards'])
        #print('avg reward: %f' % np.mean(reward_array[reward_array != 0]))
        '''
        print('after episode %d, avg reward: %10.7f, accumulated avg reward: %f, loss: %f, successes: %d' % (self.eps_counter, np.mean(discounted_reward), np.mean(self.reward_queue), loss, np.sum(reward_array > 0)))
        if self.eps_counter % self.config["save_step"] == 0:
            self.saver.save(self.sess, self.save_file + '-%d' % self.eps_counter)
        '''
