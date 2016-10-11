import gym, os
import numpy as np
import agents
from worlds import init_world

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = {
        'env': 'Pong-v0',
        'MAX_EPISODES': 10000,
        'MAX_STEPS': 1000,
        'render': False
        }

env = init_world(config['env'])
render = config['render']
MAX_EPISODES = config['MAX_EPISODES']
MAX_STEPS    = config['MAX_STEPS']

agent = agents.PGAgent(observation_dims=(env.observation_dims,), 
        action_dim=env.action_dim, eps_end=env.eps_end())

for i_eps in xrange(MAX_EPISODES):
    total_rewards = 0
    state = env.reset()

    agent.init_state(state)
    for t in xrange(MAX_STEPS):
        if render: env.render()
        action = agent.action()
        next_state, reward, done, _ = env.step(action)
        agent.experience(state, reward)
        if done: break


