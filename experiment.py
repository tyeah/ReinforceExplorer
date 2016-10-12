import gym, os, json, argparse
import numpy as np
from agents import init_agent
from worlds import init_world

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1')
parser.add_argument('-c', '--config', type=str, default='configs/pg_pong.json')
parser.add_argument('-w', '--weights', type=str, default=None)
parser.add_argument('-s', '--save_dir', type=str, default=None)
parser.add_argument('-r', '--render', action='store_true', default=False)
parser.add_argument('-l', '--learning', action='store_true', default=True)
args = parser.parse_args()

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

env = init_world(config['env'])
MAX_EPISODES = config['MAX_EPISODES']
MAX_STEPS    = config['MAX_STEPS']

agent = init_agent(config['agent'])(observation_dims=(env.observation_dims,), 
        action_dim=env.action_dim, eps_end=env.eps_end(), 
        learning=learning, config=config)

for i_eps in xrange(MAX_EPISODES):
    total_rewards = 0
    state = env.reset()

    agent.init_state(state)
    for t in xrange(MAX_STEPS):
        if render: env.render()
        action = agent.action()
        next_state, reward, done, _ = env.step(action)
        agent.experience(state, reward, done)
        if done: break
