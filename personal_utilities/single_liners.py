import numpy as np
import scipy as sp
import scipy.signal
import warnings

def even_it(x):
    return 2*int(x//2)

def reshape_to_vec(x):
    """
    This is a rather useful function for LinearOperators
    to ensure they work correctly when x is shape=(n,1) or (n,)
    """
    return x.reshape(x.shape[0]) if len(x.shape) == 2 else x

def my_resample(f, N):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sp.signal.resample(f, N)
    return out
