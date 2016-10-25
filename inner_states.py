import tensorflow as tf
import numpy as np
from collections import deque
import utils

'''
Representation of agent's inner state. It can be single state vector / 
multiple frames / rnn hidden state, etc
'''

def init_inner_state(state, **params):
    inner_states = {
            'inner_state': InnerState,
            'multi_step_inner_state': MultiStepInnerState
            }
    return inner_states[params['name']](state, **params)

class InnerState(object):
    '''
    Base class
    '''
    def __init__(self, state, **kwargs):
        self.current_state = utils.to_list(state)
        self.old_state = utils.to_list(state)

    def update(self, state):
        self.old_state = self.current_state
        self.current_state = utils.to_list(state)

    def get_current_state(self):
        return self.current_state

class MultiStepInnerState(object):
    '''
    Keep Multiple Steps in the Inner State
    '''
    def __init__(self, state, num_steps, **kwargs):
        state_list = utils.to_list(state)
        self.num_steps = num_steps
        self.range_step = range(num_steps)
        self.num_states = len(state_list)
        self.range_state = range(self.num_states)

        self.current_state = [deque(maxlen=num_steps) for _ in self.range_state]
        self.old_state = [deque(maxlen=num_steps) for _ in self.range_state]
        for _ in self.range_step:
            for qidx in self.range_state:
                self.current_state[qidx].append(state)
                self.old_state[qidx].append(state)
        self.current_state.append(state_list)
        #self.num_states = len(current_state[-1]) if utils.multi_state(state) else 1
        self.num_states = len(self.current_state[-1])
        self.state_dims = [len(s.shape) for s in state_list]

    def update_deque(self, qidx, state_list):
        self.old_state[qidx].append(self.current_state[qidx][-1])
        self.current_state[qidx].append(state_list[qidx])

    def update(self, state):
        state_list = utils.to_list(state)
        for qidx, s in enumerate(state_list):
            self.update_deque(qidx, state_list)

    def get_current_state(self):
        #cs = [np.array(self.current_state[qidx]) for qidx in self.range_state]
        cs = [np.concatenate(self.current_state[qidx], axis=self.state_dims[qidx]-1) for qidx in self.range_state]
        #cs = [np.rollaxis(np.array(self.current_state[qidx]), axis=self.state_dims[qidx]) for qidx in self.range_state]
        return cs

    def get_old_state(self):
        old_state = [np.concatenate(self.old_state[qidx], axis=self.state_dims[qidx]-1) for qidx in self.range_state]
        return old_state
