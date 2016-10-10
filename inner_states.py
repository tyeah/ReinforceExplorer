import tensorflow as tf
import numpy as np

'''
Representation of agent's inner state. It can be single state vector / 
multiple frames / rnn hidden state, etc
'''

class InnerState(object):
    '''
    Base class
    '''
    def __init__(self, state):
        if type(state) not in(list, tuple):
            self.signle_state = True
        else:
            self.single_state = False
        if self.signle_state:
            self.current_state = [state]
            self.old_state = [state]
        else:
            self.current_state = state
            self.old_state = state

    def update(self, state):
        self.old_state = self.current_state
        if self.signle_state:
            self.old_state = [state]
        else:
            self.old_state = state
