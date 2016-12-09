import gym, os, sys, json, argparse
import numpy as np
from agents import init_agent
from worlds import init_world
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1')
parser.add_argument('-c', '--config', type=str, default='configs/ddpgcont_func.json')
parser.add_argument('-w', '--weights', type=str, default=None)
parser.add_argument('-s', '--save_dir', type=str, default=None)
parser.add_argument('-r', '--render', action='store_true', default=False)
parser.add_argument('-l', '--learning', action='store_true', default=True)
args = parser.parse_args()

if float(args.gpu) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = json.load(open(args.config))
print('loaded config file: %s' % args.config)
render = args.render
learning = args.learning

if args.save_dir is None:
    args.save_dir = 'weights/' + os.path.basename(os.path.splitext(args.config)[0])
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
json.dump(config, open(args.save_dir + '/config.json', 'w'), indent=4)
config['weights'] = args.weights
config['save_dir'] = args.save_dir

if 'env_config' not in config:
    config['env_config'] = {}
env = init_world(config['env'], **config['env_config'])

MAX_EPISODES = config['MAX_EPISODES']
MAX_STEPS    = config['MAX_STEPS']

EPISODES_BEFORE_RESET = config['EPISODES_BEFORE_RESET']
agent = init_agent(config['agent'])(observation_dims=(env.observation_dims,), 
        action_dim=env.action_dim, eps_end=env.eps_end(), 
        learning=learning, config=config)

avg_rewards = deque(maxlen=100)
no_reward_since = 0
log_file = open('log.txt', 'w')
try:
    for i_eps in xrange(MAX_EPISODES):
        total_rewards = 0
        state = env.reset()
        print("start value: %f" % env.last_value)

        agent.init_state(state)
        acc_rewards = 0
        for t in xrange(MAX_STEPS):
            if render: env.render()
            #action = agent.action()
            action = agent.action() * 0.001
            #print action.shape, type(action)
            #print state
            #print type(state), state.shape
            #print action
            #print state
            #sys.exit()
            #action = 0.001 * state.reshape(-1)
            next_state, reward, done, _ = env.step(action)
            #print env.last_value
            #print reward, done, action
            acc_rewards += reward
            #reward = -10 if done else 0.1
            #reward = 5.0 if done else -0.1
            agent.experience(next_state, reward, done)
            if done: break

        final_value = env.last_value
        if not done and final_value > 100:
            no_reward_since += 1
            if no_reward_since >= EPISODES_BEFORE_RESET:
                agent.reset_model()
                no_reward_since = 0
                avg_rewards = deque(maxlen=100)
                env = init_world(config['env'], **config['env_config'])
                continue
        else:
            no_reward_since = 0
        avg_rewards.append(acc_rewards)
        print("episode %d, episode reward: %f, mean reward: %f, num steps: %d, final value: %f" % (i_eps, acc_rewards, np.mean(avg_rewards), t + 1, env.last_value))
        log_file.write('%f\n' % np.mean(avg_rewards))
except KeyboardInterrupt:
    print('KeyboardInterrupt')
    log_file.close()
