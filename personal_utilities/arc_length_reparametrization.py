import numpy as np
import fast_interp
import scipy as sp
import scipy.integrate

from personal_utilities.nufft_interpolation import nufft_interpolation1d
from personal_utilities.newton_solvers import VectorNewton
from personal_utilities.fourier_filters import SimpleFourierFilter

def _setup(n):
    dt = 2*np.pi/n
    tv = np.arange(n)*dt
    ik = 1j*np.fft.fftfreq(n, dt/(2*np.pi))
    ik[int(n/2)] = 0.0
    ikr = ik.copy()
    ikr[0] = np.Inf
    ikr[int(n/2)] = np.Inf
    return dt, tv, ik, ikr

def _get_speed(x, y, ik):
    xh = np.fft.fft(x)
    yh = np.fft.fft(y)
    xhr = xh.copy()
    yhr = yh.copy()
    xhr[int(2*xh.size/3):int(4*xh.size/3)] = 0
    yhr[int(2*yh.size/3):int(4*yh.size/3)] = 0
    xp = np.fft.ifft(xhr*ik).real
    yp = np.fft.ifft(yhr*ik).real
    sd = np.hypot(xp, yp)
    return xh, yh, sd

def _null_filter_function(f):
    return f

def fractional_fourier_filter(f, fraction):
    fh = np.fft.rfft(f)
    fh[int(fraction*fh.shape[0]):] = 0.0
    return np.fft.irfft(fh)

def old_arc_length_parameterize(x, y, tol=1e-14, filter_function=None, return_t=False):
    """
    Reparametrize the periodic curve defined by (x, y)

    Note that although this will solve the Newton problem to tol, the actual
    parameterization will not be arclength to that value unless the speed of
    the parameterization is resolved to that tolerance!  Resolving the speed
    may take significantly more resolution than the coordinates (x, y)

    Because smooth functions may have high-frequency fourier noise added by
    the reparametrization, one may consider adding a filter_function, to control
    high-frequency noise in the reparametrized (x, y)
    """
    if filter_function is None: filter_function = _null_filter_function
    if type(filter_function) is float:
        filter_function = SimpleFourierFilter()
    n = x.shape[0]
    dt, tv, ik, ikr = _setup(n)
    xh, yh, sd = _get_speed(x, y, ik)
    # total arc-length
    al = np.sum(sd)*dt
    # rescaled speed
    asd = sd*2*np.pi/al
    # fourier transform for speed and periodized arc-length
    pal_hat = np.fft.fft(asd)/ikr
    sd_hat = np.fft.fft(asd)
    # functions for newton solve to get arc-length coordinates
    def al_func(s):
        f = nufft_interpolation1d(s, pal_hat)
        return f + s - tv
    def J_func(s):
        return nufft_interpolation1d(s, sd_hat)
    # run Newton solver
    solver = VectorNewton(al_func, J_func)  
    snew = solver.solve(tv, tol=tol, verbose=False)
    # get the new x and y coordinates
    x = nufft_interpolation1d(snew, xh)
    y = nufft_interpolation1d(snew, yh)
    if return_t:
        return filter_function(x), filter_function(y), snew
    else:
        return filter_function(x), filter_function(y)

def arc_length_parameterize(x, y, tol=1e-14, filter_fraction=0.9, return_t=False):
    """
    Reparametrize the periodic curve defined by (x, y)

    Note that although this will solve the Newton problem to tol, the actual
    parameterization will not be arclength to that value unless the speed of
    the parameterization is resolved to that tolerance!  Resolving the speed
    may take significantly more resolution than the coordinates (x, y)
    """
    n = x.shape[0]
    dt, tv, ik, ikr = _setup(n)
    xh, yh, sd = _get_speed(x, y, ik)
    # total arc-length
    al = np.sum(sd)*dt
    # rescaled speed
    asd = sd*2*np.pi/al
    # fourier transform for speed and periodized arc-length
    pal_hat = np.fft.fft(asd)/ikr
    sd_hat = np.fft.fft(asd)
    # functions for newton solve to get arc-length coordinates
    def al_func(s):
        f = nufft_interpolation1d(s, pal_hat)
        return f + s - tv
    def J_func(s):
        return nufft_interpolation1d(s, sd_hat)
    # run Newton solver
    solver = VectorNewton(al_func, J_func)  
    snew = solver.solve(tv, tol=tol, verbose=False)
    if False:
        x = nufft_interpolation1d(snew, xh)
        y = nufft_interpolation1d(snew, yh)
        filt = lambda x: fractional_fourier_filter(x, filter_fraction)
        if return_t:
            return filt(x), filt(y), snew
        else:
            return filt(x), filt(y)
    else:
        snew = tv+fractional_fourier_filter(snew-tv, filter_fraction)
        x = nufft_interpolation1d(snew, xh)
        y = nufft_interpolation1d(snew, yh)
        if return_t:
            return x, y, snew
        else:
            return x, y


