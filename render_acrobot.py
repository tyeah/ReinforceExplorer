import gym

env = gym.make("Acrobot-v1")
env.reset()
while True:
    env.render()
    env.step(0)
