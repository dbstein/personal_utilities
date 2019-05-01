import numpy as np

from .transformations import affine_transformation

def get_chebyshev_nodes(lb, ub, order):
    """
    Provides chebyshev quadratures nodes
    scaled to live on the interval [lb, ub], of specified order
    The nodes are reversed from traditional chebyshev nodes
        (so that the lowest valued node comes first)
    Returns:
        unscaled nodes
        scaled nodes
        scaling ratio
    """
    xc, _ = np.polynomial.chebyshev.chebgauss(order)
    x, rat = affine_transformation(xc[::-1], -1, 1, lb, ub, return_ratio=True)
    return xc[::-1], x, rat

class ChebyshevInterpolater(object):
    def __init__(self, x, data):
        self.x = x
        self.n = self.x.shape[0]
        self.data = data
        self.coefs = np.polynomial.chebyshev.chebfit(self.x, self.data, self.n-1)
    def __call__(self, x_out):
        return np.polynomial.chebyshev.chebval(x_out, self.coefs)
