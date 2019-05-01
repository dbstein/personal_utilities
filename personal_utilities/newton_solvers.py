import numpy as np

class VectorNewton(object):
    """
    Vectorized Newton solver
    """
    def __init__(self, f, J):
        self.f = f
        self.J = J
    def solve(self, x0, tol=1e-10, verbose=False):
        x = x0.copy()
        y = self.f(x)
        r = np.abs(y).max()
        i = 0
        while r > tol:
            i += 1
            if verbose:
                print('Residual at iteration', i, 'is: {:0.2e}'.format(r))
            d = -y/self.J(x)
            l = 1.0
            while True:
                if verbose:
                    print('   Line factor is: {:0.5f}'.format(l))
                xn = x + l*d
                yn = self.f(xn)
                rn = np.abs(yn).max()
                if (rn < (1-0.5*l)*r) or l < 1e-4:
                    x = xn
                    y = yn
                    r = rn
                    break
                l *= 0.5
        return x
