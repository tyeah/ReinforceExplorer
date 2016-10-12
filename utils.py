import numpy as np

def multi_state(x):
    return hasattr(x, '__iter__') and type(x) != np.ndarray

def to_list(x):
    return x if (hasattr(x, '__iter__') and type(x) != np.ndarray) else [x]
