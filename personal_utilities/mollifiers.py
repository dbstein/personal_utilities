import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.signal
try:
    import fast_interp
except:
    pass

class Mollifier(object):
    """
    Constructs mollification 'step' and 'bump' functions
    from a prolate spheroidal wavefunction
    """
    def __init__(self, r, N=1000):
        self.r = r
        self.N = N
        try:
            import fast_interp
            self.backend = 'fast_interp'
        except:
            self.backend = 'scipy'
        self._construct()
    def bump(self, x, check_bounds=True):
        if check_bounds:
            out = np.zeros(self._get_shape(x), dtype=float)
            good = np.logical_and(x > -1, x < 1)
            out[good] = self._bump(x[good])
        else:
            out = self._bump(x)
        return out
    def step(self, x, check_bounds=True):
        if check_bounds:
            out = np.zeros(self._get_shape(x), dtype=float)
            low =  x <= -1
            high = x >=  1
            good = np.logical_not(np.logical_or(low, high))
            out[good] = self._step(x[good])
            out[high] = 1.0
        else:
            out = self._step(x)
        return out
    def plot(self, ax, what='both', *args, **kwargs):
        if what == 'both' or what == 'bump':
            ax.plot(self._x, self._bump(self._x), *args, **kwargs)
        if what == 'both' or what == 'step':
            ax.plot(self._x, self._step(self._x), *args, **kwargs)
    def _interpolate(self, f):
        if self.backend == 'scipy':
            return sp.interpolate.InterpolatedUnivariateSpline(self._x, f, k=5, ext='zeros')
        else:
            return fast_interp.interp1d(-1, 1, self._h, f, k=5)
    def _construct(self):
        self._x, self._h = np.linspace(-1, 1, self.N, endpoint=True, retstep=True)
        w = sp.signal.slepian(self.N, float(self.r)/self.N)
        self._bump = self._interpolate(w)
        w_int = np.zeros(self.N, dtype=float)
        for i in range(1, self.N):
            w_int[i] = w_int[i-1] + \
                sp.integrate.quad(self._bump, self._x[i-1], self._x[i], epsabs=1e-14)[0]
        w_int /= w_int[-1]
        self._step = self._interpolate(w_int)
    def _get_shape(self, x):
        try:
            shape = x.shape
        except:
            shape = 1
        return shape
