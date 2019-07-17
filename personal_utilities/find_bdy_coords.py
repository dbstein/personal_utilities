import numpy as np
import fast_interp
from .single_liners import my_resample

def find_boundary_normal_coordinates(bdy, x, y, newton_tol, guess_ind=None, verbose=False):
    """
    Find using the coordinates:
    x = X - r n_x
    y = Y + r n_y
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    xp = np.fft.ifft(np.fft.fft(bdy.x)*bdy.ik).real
    yp = np.fft.ifft(np.fft.fft(bdy.y)*bdy.ik).real
    nxp = np.fft.ifft(np.fft.fft(bdy.normal_x)*bdy.ik).real
    nyp = np.fft.ifft(np.fft.fft(bdy.normal_y)*bdy.ik).real
    def interp(f):
        return fast_interp.interp1d(0.0, 2*np.pi, bdy.dt, f, k=5, p=True)
    nx_i =  interp(bdy.normal_x)
    ny_i =  interp(bdy.normal_y)
    nxp_i = interp(nxp)
    nyp_i = interp(nyp)
    x_i =   interp(bdy.x)
    y_i =   interp(bdy.y)
    xp_i =  interp(xp)
    yp_i =  interp(yp)

    def f(t, r):
        x = x_i(t) + r*nx_i(t)
        y = y_i(t) + r*ny_i(t)
        return x, y
    def Jac(t, r):
        dxdt = xp_i(t) + r*nxp_i(t)
        dydt = yp_i(t) + r*nyp_i(t)
        dxdr = nx_i(t)
        dydr = ny_i(t)
        J = np.zeros((t.shape[0],2,2),dtype=float)
        J[:,0,0] = dxdt
        J[:,0,1] = dydt
        J[:,1,0] = dxdr
        J[:,1,1] = dydr
        return J.transpose((0,2,1))

    if guess_ind is None:
        # brute force find of guess_inds
        xd = x - bdy.x[:,None]
        yd = y - bdy.y[:,None]
        dd = xd**2 + yd**2
        guess_ind = dd.argmin(axis=0)

    t = bdy.t[guess_ind]
    xdg = x - bdy.x[guess_ind]
    ydg = y - bdy.y[guess_ind]
    r = np.sqrt(xdg**2 + ydg**2)
    xo, yo = f(t, r)
    remx = xo - x
    remy = yo - y
    rem = np.abs(np.sqrt(remx**2 + remy**2)).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(rem))
    while rem > newton_tol:
        J = Jac(t, r)
        delt = -np.linalg.solve(J, np.column_stack([remx, remy]))
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt[:,0], r + line_factor*delt[:,1]
            xo, yo = f(t_new, r_new)
            remx = xo - x
            remy = yo - y
            rem_new = np.sqrt(remx**2 + remy**2).max()
            # print line_factor
            if (rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4:
                t = t_new
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
        if verbose:
            print('Newton tol: {:0.2e}'.format(rem))
    # put theta back in [0, 2 pi]
    t[t < 0] += 2*np.pi
    t[t > 2*np.pi] -= 2*np.pi
    return t, r

def find_boundary_alpha_coordinates(bdy, x, y, newton_tol, guess_ind=None, verbose=False):
    """
    Find using the simple coordinates:
    x = X - r Y_s
    y = Y + r X_s
    (non-normalized normal coordinates)
    """
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    xh = np.fft.fft(bdy.x)
    yh = np.fft.fft(bdy.y)
    xp = np.fft.ifft(xh*bdy.ik).real
    yp = np.fft.ifft(yh*bdy.ik).real
    xpp = np.fft.ifft(xh*bdy.ik**2).real
    ypp = np.fft.ifft(yh*bdy.ik**2).real
    def interp(f):
        ff = my_resample(f, 10*bdy.N)
        return fast_interp.interp1d(0.0, 2*np.pi, bdy.dt/10, ff, k=5, p=True)
    x_i =   interp(bdy.x)
    y_i =   interp(bdy.y)
    xp_i =  interp(xp)
    yp_i =  interp(yp)
    xpp_i =  interp(xpp)
    ypp_i =  interp(ypp)

    def f(t, r):
        x = x_i(t) - r*yp_i(t)
        y = y_i(t) + r*xp_i(t)
        return x, y
    def Jac(t, r):
        dxdt = xp_i(t) - r*ypp_i(t)
        dydt = yp_i(t) + r*xpp_i(t)
        dxdr = -yp_i(t)
        dydr = xp_i(t)
        J = np.zeros((t.shape[0],2,2),dtype=float)
        J[:,0,0] = dxdt
        J[:,0,1] = dydt
        J[:,1,0] = dxdr
        J[:,1,1] = dydr
        return J.transpose((0,2,1))

    if guess_ind is None:
        # brute force find of guess_inds
        xd = x - bdy.x[:,None]
        yd = y - bdy.y[:,None]
        dd = xd**2 + yd**2
        guess_ind = dd.argmin(axis=0)

    t = bdy.t[guess_ind]
    xdg = x - bdy.x[guess_ind]
    ydg = y - bdy.y[guess_ind]
    r = np.sqrt(xdg**2 + ydg**2)/bdy.speed[guess_ind]
    xo, yo = f(t, r)
    remx = xo - x
    remy = yo - y
    rem = np.abs(np.sqrt(remx**2 + remy**2)).max()
    if verbose:
        print('Newton tol: {:0.2e}'.format(rem))
    while rem > newton_tol:
        J = Jac(t, r)
        delt = -np.linalg.solve(J, np.column_stack([remx, remy]))
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt[:,0], r + line_factor*delt[:,1]
            xo, yo = f(t_new, r_new)
            remx = xo - x
            remy = yo - y
            rem_new = np.sqrt(remx**2 + remy**2).max()
            # print line_factor
            if (rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4:
                t = t_new
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
        if verbose:
            print('Newton tol: {:0.2e}'.format(rem))
    # put theta back in [0, 2 pi]
    t[t < 0] += 2*np.pi
    t[t > 2*np.pi] -= 2*np.pi
    return t, r

