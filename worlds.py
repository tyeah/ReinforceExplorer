import tensorflow as tf
import numpy as np
import gym

def init_world(name):
    worlds = {
            'gym_world': GymWorld,
            'Pong-v0': PongWorld
            }
    if name not in worlds:
        return worlds[gym_world](name)
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

class PongWorld(GymWorld):
    def __init__(self, name):
        super(PongWorld, self).__init__(name)

    def eps_end(self):
        def ee(reward, state=None):
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