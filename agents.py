import tensorflow as tf
import numpy as np
from inner_states import init_inner_state, InnerState
from estimators import Estimator
from collections import deque
import os, sys
from copy import deepcopy


def init_agent(agent_name):
    agents = {
            'naive': Agent,
            'policy_gradient': PGAgent,
            'actor_critic': ACAgent,
            'ddpg': DDPGAgent,
            'ddpg_cont': DDPGContAgent,
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
        '''
        if self.config["clip_norm"] <= 0:
            grads = tf.gradients(self.loss, tvars)
        else:
            grads, norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config["clip_norm"])
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        '''
        gradients = tf.gradients(self.loss, tvars)
        if self.config['clip_norm'] >= 0:
            for i, (grad, var) in enumerate(zip(gradients, tvars)):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self.config['clip_norm']), var) 
        self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)

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

    def add_reg(self, variables):
        self.reg_loss = 0
        if 'l2' in self.config and self.config['l2'] > 0:
            self.reg_loss += self.config['l2'] * tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
        if 'l1' in self.config and self.config['l1'] > 0:
            self.reg_loss += self.config['l1'] * tf.reduce_sum([tf.nn.l1_loss(v) for v in variables])

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
        #with tf.variable_scope('Agent'):
        self.build_model()
        self.saver = tf.train.Saver()
        self.init_weights()
        '''
        self.add_summary()
        '''
    def init_weights(self):
        if self.config["weights"] is not None:
            try:
                self.saver.restore(self.sess, self.config["weights"])
                print "loaded weights:", self.config['weights']
            except tf.python.errors.NotFoundError:
                ckpt_reader = tf.train.NewCheckpointReader(self.config['weights'])
                var_list_restore = {v.name.split(':')[0]: v for
                        v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 
                        ckpt_reader.has_tensor(v.name.split(':')[0]) 
                        and not v.name.startswith('World')}
                var_list_init = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES)
                        if not ckpt_reader.has_tensor(v.name.split(':')[0]) 
                        and not v.name.startswith('World')]
                if len(var_list_restore) > 0:
                    loader = tf.train.Saver(var_list_restore)
                    loader.restore(self.sess, self.config['weights'])
                print "loaded part of weights:", self.config['weights']
                print var_list_restore.keys()
                if len(var_list_init) > 0:
                    init = tf.initialize_variables(var_list_init)
                    self.sess.run(init)

        else:
            self.sess.run(tf.initialize_all_variables())

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od[:-1] + (od[-1] * self.config["inner_state_params"].get("num_steps", 1),)) for od in self.observation_dims]
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

        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * 
                (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), 
                    tf.float32)), self.config["min_lr"])

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
        self.reinforce_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.action_scores , self.t_action) * t_discounted_reward_reparamed)
        self.add_reg(policy_network_variables)
        self.loss = tf.identity(self.reinforce_loss)
        self.loss += self.reg_loss
        self.build_train()

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

    def reset_model(self):
        print("reset model")
        self.sess.run(self.global_step.assign(0))
        self.reset_buffer()
        self.sess.run(tf.initialize_all_variables())

    def init_state(self, state):
        self.inner_state = init_inner_state(state, **self.config['inner_state_params'])
        current_inner_state = self.inner_state.get_current_state()
        for sidx, s in enumerate(self.rollouts['states']):
            s.append(current_inner_state[sidx])

    def action(self):
        # tack action based on current state
        action = self.sess.run(self.action_sampler, feed_dict=dict([(s_t, [s]) for s_t, s in zip(self.t_state, self.inner_state.get_current_state())]))[0]
        self.rollouts['actions'].append(action)
        return action

    def experience(self, state, reward, done):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        # TODO: change the stored states as inner states
        self.rollouts['rewards'].append(reward)
        self.rollouts['eps_end_masks'].append(self.eps_end(done, reward, state))
        if not self.eps_end(done, reward, state):
            current_inner_state = self.inner_state.get_current_state()
            for sidx, s in enumerate(self.rollouts['states']):
                s.append(current_inner_state[sidx])
        # TODO: ansemble states, discounti rewards, actions, do update

        if self.cond():
            self.eps_counter += np.sum(self.rollouts['eps_end_masks'])
            if self.learning:
                self.update()
            self.reset_buffer()
        return 0

    def gen_feed(self, discounted_reward):
        num_batches = int(np.ceil(len(self.rollouts['rewards']) * 1.0 / self.config["batch_size"]))
        ########
        '''
        discounted_reward = discounted_reward[:-1]
        self.rollouts['actions'] = self.rollouts['actions'][:-1]
        for i, v in enumerate(self.rollouts['states']):
            self.rollouts['states'][i] = self.rollouts['states'][i][:-1]
        '''
        ########
        #num_batches = 1
        for i in xrange(num_batches):
            feed = dict([(s_t, np.array(s[i:(i+self.config["batch_size"])])) for s_t, s in zip(self.t_state, self.rollouts['states'])] +
                    [(self.t_discounted_reward, np.array(discounted_reward[i:(i+self.config["batch_size"])])), (self.t_action, np.array(self.rollouts['actions'][i:(i+self.config["batch_size"])]))])
            yield feed



    def compute_discount_rewards(self):
        discounted_reward = np.zeros_like(self.rollouts['rewards'])
        for i in reversed(xrange(0, len(self.rollouts['rewards']))):
            if self.rollouts['eps_end_masks'][i]:
                dr = 0
            dr = dr * self.config["discount_rate"] + self.rollouts['rewards'][i]
            discounted_reward[i] = dr
        return discounted_reward

    def update(self):
        # learn from experiences
        discounted_reward = self.compute_discount_rewards()
            
        # TODO: when to update, how to set batch, non-iid sgd, loss func
        self.reward_queue.append(np.mean(discounted_reward))
        discounted_reward -= np.mean(discounted_reward)
        #discounted_reward /= np.std(discounted_reward)
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


class ACAgent(PGAgent):
    '''
    Actor Critic
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
        super(ACAgent, self).__init__(observation_dims, action_dim, config, eps_end, learning)

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od[:-1] + (od[-1] * self.config["inner_state_params"].get("num_steps", 1),)) for od in self.observation_dims]
        self.t_action = tf.placeholder(dtype=tf.int32, shape=(None,)) # for discrete action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        '''
        self.exp_rate = self.config["init_exp_rate"] * (self.config["anneal_base_exp"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_exp"]), tf.float32))
        '''

        self.exp_rate = tf.cast(tf.maximum(self.config['anneal_step_exp'] - self.global_step, 0), tf.float32) / (1.0 * self.config['anneal_step_exp']) * (self.config["init_exp_rate"] - self.config["min_exp"]) + self.config["min_exp"]

        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * 
                (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), 
                    tf.float32)), self.config["min_lr"])

        self.actor = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state, num_out=self.action_dim, 
                scope='policy_network', 
                **self.config['estimator_params']['policy_network'])
        self.critic = Estimator(self.config['estimator_params']
                ['value_network']['name']).get_estimator(
                inputs=self.t_state, num_out=1, 
                scope='value_network', 
                **self.config['estimator_params']['value_network'])
        policy_network_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")
        self.action_probs = tf.nn.softmax(self.actor)
        self.explore = tf.less(tf.random_uniform([batch_size]), self.exp_rate)
        self.action_sampler = tf.select(self.explore, 
                tf.multinomial(random_action_probs, num_samples=1),
                tf.multinomial(self.action_probs, num_samples=1))
        # TODO: seed?
        # TODO: how to measure global_step?
        value_network_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="value_network")

        advantage = self.t_discounted_reward - self.critic
        self.actor_loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.actor, self.t_action) * advantage)
        self.critic_loss = tf.reduce_mean(tf.square(self.t_discounted_reward - self.critic))
        self.add_reg(policy_network_variables)
        self.add_reg(value_network_variables)
        self.loss = (self.actor_loss + self.critic_loss)
        self.loss += self.reg_loss
        self.build_train()

    def update(self):
        # learn from experiences
        discounted_reward = self.compute_discount_rewards()
            
        # TODO: when to update, how to set batch, non-iid sgd, loss func
        self.reward_queue.append(np.mean(discounted_reward))
        #discounted_reward -= np.mean(discounted_reward)
        #discounted_reward /= np.std(discounted_reward)

        feeds = self.gen_feed(discounted_reward)
        for feed in feeds:
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)


class DDPGAgent(PGAgent):
    '''
    DDPG actor critic
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
        super(DDPGAgent, self).__init__(observation_dims, action_dim, config, eps_end, learning)
        self.init_target()
        self.cond = lambda: self.acc_memory_size >= self.config['batch_size']

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, shape=(None,) + od[:-1] + (od[-1] * self.config["inner_state_params"].get("num_steps", 1),)) for od in self.observation_dims]
        self.t_state_new = [tf.placeholder(dtype=tf.float32, shape=(None,) + od[:-1] + (od[-1] * self.config["inner_state_params"].get("num_steps", 1),)) for od in self.observation_dims]
        self.t_action = tf.placeholder(dtype=tf.int32, shape=(None,)) # for discrete action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.t_reward = tf.placeholder(dtype=tf.float32, shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        '''
        self.exp_rate = self.config["init_exp_rate"] * (self.config["anneal_base_exp"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_exp"]), tf.float32))
        '''

        self.exp_rate = tf.cast(tf.maximum(self.config['anneal_step_exp'] - self.global_step, 0), tf.float32) / (1.0 * self.config['anneal_step_exp']) * (self.config["init_exp_rate"] - self.config["min_exp"]) + self.config["min_exp"]

        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * 
                (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), 
                    tf.float32)), self.config["min_lr"])

        self.actor = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state, num_out=self.action_dim, 
                scope='actor', 
                **self.config['estimator_params']['policy_network'])
        self.critic = Estimator(self.config['estimator_params']
                ['value_network']['name']).get_estimator(
                inputs=self.t_state, actions=self.t_action, num_out=1, 
                scope='critic', 
                **self.config['estimator_params']['value_network'])

        self.action_sampler_deterministic = tf.argmax(self.actor, dimension=1)
        self.action_probs = tf.nn.softmax(self.actor)
        self.explore = tf.less(tf.random_uniform([batch_size]), self.exp_rate)
        self.action_sampler = tf.select(self.explore, 
                tf.multinomial(random_action_probs, num_samples=1),
                tf.multinomial(self.action_probs, num_samples=1))

        actor_target_config = deepcopy(self.config['estimator_params']['policy_network'])
        actor_target_config['trainable'] = False
        self.actor_target = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state_new, num_out=self.action_dim, 
                scope='actor_target',
                **actor_target_config)
        critic_target_config = deepcopy(self.config['estimator_params']['policy_network'])
        critic_target_config['trainable'] = False
        self.critic_target = Estimator(self.config['estimator_params']
                ['value_network']['name']).get_estimator(
                inputs=self.t_state_new, actions=self.action_sampler_deterministic, num_out=1, 
                scope='critic_target',
                **critic_target_config)

        actor_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="actor")])
        critic_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="critic")])
        actor_target_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="actor_target")])
        critic_target_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="critic_target")])

        self.target_init_ops = []
        for k, v in actor_variables.iteritems():
            self.target_init_ops.append(actor_target_variables[k].assign(v))
        for k, v in critic_variables.iteritems():
            self.target_init_ops.append(critic_target_variables[k].assign(v))
        self.target_update_ops = []
        tau = self.config["tau"]
        for k, v in actor_variables.iteritems():
            self.target_update_ops.append(actor_target_variables[k].assign(
                tau * v + (1.0 - tau) * actor_target_variables[k]))
        for k, v in critic_variables.iteritems():
            self.target_update_ops.append(critic_target_variables[k].assign(
                tau * v + (1.0 - tau) * critic_target_variables[k]))

        self.actor_loss = tf.reduce_mean(self.critic_target)
        self.target = self.t_reward + self.config["discount_rate"] * self.critic_target
        self.critic_loss = tf.reduce_mean(self.target - self.critic)

        self.add_reg(actor_variables.values())
        self.add_reg(critic_variables.values())
        self.loss = (self.actor_loss + self.critic_loss)
        self.loss += self.reg_loss
        self.build_train()

    def build_train(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                decay=0.9)
        tvars = [v for v in self.loss.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if not v.name.startswith('World')]
        if self.config["clip_norm"] <= 0:
            grads = tf.gradients(self.loss, tvars)
        else:
            grads, norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config["clip_norm"])
            print self.loss, self.loss.get_shape(), len(tvars), type(self.loss)
        for i, g in enumerate(grads):
            if g is not None:
                print g.get_shape(), tvars[i].get_shape(), tvars[i].name
            else:
                print None, tvars[i].get_shape(), tvars[i].name
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def gen_feed(self):
        batch_indices = np.random.randint(0, self.acc_memory_size, self.config['batch_size'])
        feed = dict([(s_t, [s[bi] for bi in batch_indices]) 
            for s_t, s in zip(self.t_state, self.rollouts['states_old'])] +
            [(s_t, [s[bi] for bi in batch_indices]) 
                for s_t, s in zip(self.t_state_new, self.rollouts['states'])])
        feed[self.t_reward] = [self.rollouts['rewards'][bi] for bi in batch_indices]
        feed[self.t_action] = [self.rollouts['actions'][bi] for bi in batch_indices]
        return feed

    def reset_buffer(self):
        self.acc_memory_size = 0
        ms = self.config['memory_size']
        deque(maxlen=ms)
        self.rollouts = {
                'states': [deque(maxlen=ms) for _ in xrange(len(self.observation_dims))],
                'states_old': [deque(maxlen=ms) for _ in xrange(len(self.observation_dims))],
                'rewards': deque(maxlen=ms),
                'actions': deque(maxlen=ms),
                }

    def init_state(self, state):
        self.inner_state = init_inner_state(state, **self.config['inner_state_params'])
        '''
        print '-' * 80
        print len(self.inner_state.current_state[0]), type(self.inner_state.current_state[0])
        print self.inner_state.current_state[0][0]
        print '-' * 80
        print self.inner_state.current_state[0][1]
        sys.exit()
        '''
        current_inner_state = self.inner_state.get_current_state()
        for sidx, s in enumerate(self.rollouts['states_old']):
            s.append(current_inner_state[sidx])

    def experience(self, state, reward, done):
        # experience current state
        # receive the reward for last action
        self.inner_state.update(state)
        # TODO: change the stored states as inner states
        self.rollouts['rewards'].append(reward)
        self.acc_memory_size = len(self.rollouts['rewards'])
        #self.rollouts['eps_end_masks'].append(self.eps_end(done, reward, state))
        current_inner_state = self.inner_state.get_current_state()
        for sidx, s in enumerate(self.rollouts['states']):
            s.append(current_inner_state[sidx])
        if not self.eps_end(done, reward, state):
            for sidx, s in enumerate(self.rollouts['states_old']):
                s.append(current_inner_state[sidx])
        # TODO: ansemble states, discounti rewards, actions, do update
        #print self.acc_memory_size, len(self.rollouts['rewards']), len(self.rollouts['actions']), len(self.rollouts['states'][0]), len(self.rollouts['states_old'][0])

        if self.cond():
            if self.learning:
                self.update()
        return 0

    def update(self):
        # learn from experiences
        feed = self.gen_feed()
        #print [v.name for v in feed]
        sys.exit()
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        self.update_target()

    def update_target(self):
        for tu_op in self.target_update_ops:
            self.sess.run(tu_op)

    def init_target(self):
        for ti_op in self.target_init_ops:
            self.sess.run(ti_op)

    def reset_model(self):
        print("reset model")
        self.sess.run(self.global_step.assign(0))
        self.reset_buffer()
        self.sess.run(tf.initialize_all_variables())
        self.init_target()


class DDPGContAgent(DDPGAgent):
    '''
    DDPG actor critic
    '''
    def __init__(self, observation_dims, action_dim, config, eps_end=None, learning=False):
        super(DDPGContAgent, self).__init__(observation_dims, action_dim, config, eps_end, learning)
        #TODO: low/high for action

    def build_model(self):
        # TODO: should we set exp rate as a placeholder?
        self.t_state = [tf.placeholder(dtype=tf.float32, 
            name='t_state_%d' % i,
            shape=(None,) + od[:-1] + (od[-1] * 
            self.config["inner_state_params"].get("num_steps", 1),)) 
            for i, od in enumerate(self.observation_dims)]
        #print self.t_state[0].get_shape(), self.observation_dims, self.action_dim
        self.t_state_new = [tf.placeholder(dtype=tf.float32, 
            name='t_state_new_%d' % i,
            shape=(None,) + od[:-1] + (od[-1] * self.config["inner_state_params"].
                get("num_steps", 1),)) for i, od in enumerate(self.observation_dims)]
        self.t_action = tf.placeholder(dtype=tf.int32, name='t_action', 
                shape=(None,) + self.action_dim) # for action space
        self.t_discounted_reward = tf.placeholder(dtype=tf.float32, 
                name='t_discounted_reward', shape=(None,))
        self.t_reward = tf.placeholder(dtype=tf.float32, 
                name='t_reward', shape=(None,))
        batch_size = tf.shape(self.t_state)[0]
        #TODO: only used for discrete action:random_action_probs = tf.fill((batch_size, self.action_dim), 1.0 / self.action_dim)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        '''
        self.exp_rate = self.config["init_exp_rate"] * (self.config["anneal_base_exp"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_exp"]), tf.float32))
        '''

        self.exp_rate = tf.cast(tf.maximum(self.config['anneal_step_exp'] - self.global_step, 0), tf.float32) / (1.0 * self.config['anneal_step_exp']) * (self.config["init_exp_rate"] - self.config["min_exp"]) + self.config["min_exp"]

        self.learning_rate = tf.maximum(self.config["init_learning_rate"] * 
                (self.config["anneal_base_lr"] ** 
                tf.cast(tf.floordiv(self.global_step, self.config["anneal_step_lr"]), 
                    tf.float32)), self.config["min_lr"])

        self.actor = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state, num_out=np.prod(self.action_dim), 
                scope='actor', 
                **self.config['estimator_params']['policy_network'])
        #print self.actor.get_shape(), (-1,) + self.action_dim
        self.actor_id = tf.identity(self.actor)
        self.actor = tf.reshape(self.actor, (-1,) + self.action_dim)
        #TODO:
        #self.action_scale = 1e-3
        self.action_scale = 1
        self.actor *= self.action_scale
        self.critic = Estimator(self.config['estimator_params']
                ['value_network']['name']).get_estimator(
                inputs=self.t_state, actions=self.t_action, num_out=1, 
                scope='critic', 
                **self.config['estimator_params']['value_network'])

        self.action_sampler_deterministic = self.actor

        self.action_sampler = self.actor + tf.random_normal(tf.shape(self.actor), stddev=self.exp_rate)

        actor_target_config = deepcopy(self.config['estimator_params']['policy_network'])
        actor_target_config['trainable'] = False
        self.actor_target = Estimator(self.config['estimator_params']
                ['policy_network']['name']).get_estimator(
                inputs=self.t_state_new, num_out=np.prod(self.action_dim), 
                scope='actor_target',
                **actor_target_config)
        #print self.actor_target.get_shape(), (-1,) + self.action_dim
        self.actor_target = tf.reshape(self.actor_target, (-1,) + self.action_dim)
        self.actor_target *= self.action_scale
        critic_target_config = deepcopy(self.config['estimator_params']['policy_network'])
        critic_target_config['trainable'] = False
        self.critic_target = Estimator(self.config['estimator_params']
                ['value_network']['name']).get_estimator(
                inputs=self.t_state_new, actions=self.action_sampler_deterministic, num_out=1, 
                scope='critic_target',
                **critic_target_config)

        actor_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="actor")])
        critic_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="critic")])
        actor_target_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="actor_target")])
        critic_target_variables = dict([('/'.join(v.name.split('/')[1:]), v) 
                for v in tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="critic_target")])

        self.target_init_ops = []
        for k, v in actor_variables.iteritems():
            self.target_init_ops.append(actor_target_variables[k].assign(v))
        for k, v in critic_variables.iteritems():
            self.target_init_ops.append(critic_target_variables[k].assign(v))
        self.target_update_ops = []
        tau = self.config["tau"]
        for k, v in actor_variables.iteritems():
            self.target_update_ops.append(actor_target_variables[k].assign(
                tau * v + (1.0 - tau) * actor_target_variables[k]))
        for k, v in critic_variables.iteritems():
            self.target_update_ops.append(critic_target_variables[k].assign(
                tau * v + (1.0 - tau) * critic_target_variables[k]))

        self.actor_loss = tf.reduce_mean(self.critic_target)
        self.target = tf.reshape(self.t_reward, \
                [-1] + [1] * (self.critic.get_shape().ndims - 1))\
                + self.config["discount_rate"] * self.critic_target
        self.critic_loss = tf.reduce_mean(self.target - self.critic)

        self.add_reg(actor_variables.values())
        self.add_reg(critic_variables.values())
        self.loss = (self.actor_loss + self.critic_loss)
        self.loss += self.reg_loss
        self.build_train()

    def update(self):
        # learn from experiences
        feed = self.gen_feed()
        '''
        for k, v in feed.iteritems():
            print k.name
            if type(v) == list:
                print v[0].shape, len(v)
        sys.exit(0)
        '''
        _, loss, global_step = self.sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed)
        #print actor.shape
        self.update_target()
        if global_step % self.config["save_step"] == 0 and global_step > 0:
            filename = self.save_file# + '-%d' % global_step
            print "save to %s" % filename
            self.saver.save(self.sess, filename)
