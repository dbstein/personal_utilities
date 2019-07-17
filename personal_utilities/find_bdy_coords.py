import numpy as np
import fast_interp

def find_boundary_coordinates(bdy, x, y, newton_tol, guess_ind=None):
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    xp = np.fft.ifft(np.fft.fft(bdy.x)*bdy.ik).real
    yp = np.fft.ifft(np.fft.fft(bdy.y)*bdy.ik).real
    nxp = np.fft.ifft(np.fft.fft(bdy.normal_x)*bdy.ik).real
    nyp = np.fft.ifft(np.fft.fft(bdy.normal_y)*bdy.ik).real
    interp = lambda f: \
        fast_interp.interp1d(0.0, 2*np.pi, bdy.dt, f, k=5, p=True)
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
    rem = np.abs(remx**2 + remy**2).max()
    while rem > newton_tol:
        J = Jac(t, r)
        delt = -np.linalg.solve(J, np.column_stack([remx, remy]))
        line_factor = 1.0
        while True:
            t_new, r_new = t + line_factor*delt[:,0], r + line_factor*delt[:,1]
            xo, yo = f(t_new, r_new)
            remx = xo - x
            remy = yo - y
            rem_new = np.abs(remx**2 + remy**2).max()
            # print line_factor
            if (rem_new < (1-0.5*line_factor)*rem) or line_factor < 1e-4:
                t = t_new
                r = r_new
                rem = rem_new
                break
            line_factor *= 0.5
    # put theta back in [0, 2 pi]
    t[t < 0] += 2*np.pi
    t[t > 2*np.pi] -= 2*np.pi
    return t, r
