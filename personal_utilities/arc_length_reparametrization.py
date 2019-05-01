import numpy as np
import fast_interp
import scipy as sp
import scipy.integrate

from .nufft_interpolation import nufft_interpolation1d
from .newton_solvers import VectorNewton

# reparameterize a curve
def arc_length_parameterize(x, y, tol=1e-8, fourier_filter=None):
    if fourier_filter is not None:
        x = fourier_filter(x)
        y = fourier_filter(y)
    n = x.shape[0]
    dt = 2*np.pi/n
    tv = np.arange(n)*dt            # t values
    k = np.fft.fftfreq(n, dt/(2*np.pi))
    ik = 1j*k
    xh = np.fft.fft(x)              # fft of x-coords
    yh = np.fft.fft(y)              # fft of y-coords
    xp = np.fft.ifft(xh*ik).real    # derivative w.r.t. implicit parameter t
    yp = np.fft.ifft(yh*ik).real    # derivative w.r.t. implicit parameter t
    sd = np.hypot(xp, yp)           # speed w.r.t. implicit parameter t
    al = np.sum(sd*dt)              # total arc-length of curve
    ad = al/(2*np.pi)
    ai = 1.0/ad
    adt = tv*ad                     # scaled t values
    # compute the cumulative arclength
    arclength = [0.0]
    speed_interpolater = fast_interp.interp1d(0.0, 2*np.pi, dt, sd, k=5, p=True)
    for i in range(1, n):
        ia = sp.integrate.quad(speed_interpolater, tv[i-1], tv[i], epsabs=1e-12)
        arclength.append(arclength[-1]+ia[0])
    arclength = np.array(arclength)
    # periodized cumulative arclength (to allow for nufft interpolation)
    pal = arclength - adt
    # ffts of the speed and the periodized arc-length
    sd_hat = np.fft.fft(sd)
    pal_hat = np.fft.fft(pal)
    # functions for newton solve to get arc-length coordinates
    def al_func(s):
        # returns int_0^s phi(t) dt - s
        f = nufft_interpolation1d(s*ai, pal_hat)
        return f + s - adt
    def J_func(s):
        # returns phi(s)
        return nufft_interpolation1d(s*ai, sd_hat)  
    solver = VectorNewton(al_func, J_func)  
    snew = solver.solve(adt, tol=tol, verbose=False)    
    # get the new x and y coordinates
    x = nufft_interpolation1d(snew*ai, xh)
    y = nufft_interpolation1d(snew*ai, yh)
    if fourier_filter is not None:
        x = fourier_filter(x)
        y = fourier_filter(y)
    return x, y

