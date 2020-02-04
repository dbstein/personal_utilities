import numpy as np

def fast_dot(M1, M2):
    """
    Specialized interface to the numpy.dot function
    This assumes that A and B are both 2D arrays (in practice)
    When A or B are represented by 1D arrays, they are assumed to reprsent
        diagonal arrays
    This function then exploits that to provide faster multiplication
    """
    if len(M1.shape) in [1, 2] and len(M2.shape) == 1:
        return M1*M2
    elif len(M1.shape) == 1 and len(M2.shape) == 2:
        return M1[:,None]*M2
    elif len(M1.shape) == 2 and len(M2.shape) == 2:
        return M1.dot(M2)
    else:
        raise Exception('fast_dot requires shapes to be 1 or 2')
