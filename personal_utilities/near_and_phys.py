import numpy as np
import numba

def find_near_numba(bx, by, xv, yv, d, xh=None, yh=None, xlb=None, ylb=None,
                                                            xsd=None, ysd=None):
    """
    Inputs:
        bx:      x-coordinates of boundary
        by:      y-coordinates of boundary
        xv:      x-values for grid coordinates
        yv:      y-values for grid coordinates
        xh:      x-grid spacing (xv[1]-xv[0] if not provided)
        yh:      y-grid spacing (yv[1]-yv[0] if not provided)
        xlb:     grid lower bound (xv[0] if not provided)
        ylb:     grid upper bound (yv[0] if not provided)
        xsd:     search distance in x direction (d//xh + 1 if not provided)
        ysd:     search distance in y direction (d//yh + 1 if not provided)
    Outputs:
        close,     bool(sh),  whether that point within d of any boundary point or not
        guess_ind, int(sh),   the index of the closest point on the boundary to that point
        closest,   float(sh), the closest distance to a boundary point
    """
    # construct outputs
    sh = (xv.shape[0], yv.shape[0])
    close     = np.zeros(sh, dtype=int)
    guess_ind = np.zeros(sh, dtype=int)   - 1
    closest   = np.zeros(sh, dtype=float) + 1e15
    # construct inputs
    if xh is None:
        xh = xv[1] - xv[0]
    if yh is None:
        yh = yv[1] - yv[0]
    if xlb is None:
        xlb = xv[0]
    if ylb is None:
        ylb = yv[0]
    if xsd is None:
        xsd = d//xh + 1
    if ysd is None:
        ysd = d//yh + 1
    d2 = d*d
    # call the search algorithm
    _find_near_numba(bx, by, xv, yv, xh, yh, xlb, ylb, xsd, ysd, d2, close, guess_ind, closest)
    # return
    return close > 0, guess_ind, closest


@numba.njit(parallel=True)
def _find_near_numba(x, y, xv, yv, xh, yh, xlb, ylb, xsd, ysd, d2, close, gi, closest):
    N = x.shape[0]
    Nx = xv.shape[0]
    Ny = yv.shape[0]
    for i in numba.prange(N):
        x_loc = (x[i] - xlb) // xh
        y_loc = (y[i] - ylb) // yh
        x_lower = max(x_loc - xsd, 0)
        x_upper = min(x_loc + xsd + 1, Nx)
        y_lower = max(y_loc - ysd, 0)
        y_upper = min(y_loc + ysd + 1, Ny)
        for j in range(x_lower, x_upper):
            for k in range(y_lower, y_upper):
                xd = xv[j] - x[i]
                yd = yv[k] - y[i]
                dist2 = xd**2 + yd**2
                close[j, k] += int(dist2 < d2)
                if dist2 < closest[j, k]:
                    closest[j, k] = dist2
                    gi[j, k] = i

def numba_find_phys(x, y, bdyx, bdyy):
    """
    Computes whether the points x, y are inside of the polygon defined by the
    x-coordinates bdyx and the y-coordinates bdyy
    The polgon is assumed not to be closed (the last point is not replicated)
    """
    inside = np.zeros(x.shape, dtype=bool)
    vecPointInPath(x.ravel(), y.ravel(), bdyx, bdyy, inside.ravel())
    return inside.reshape(x.shape)
@numba.njit(parallel=True)
def vecPointInPath(x, y, polyx, polyy, inside):
    m = x.shape[0]
    for i in numba.prange(m):
        inside[i] = isPointInPath(x[i], y[i], polyx, polyy)
@numba.njit
def isPointInPath(x, y, polyx, polyy):
    num = polyx.shape[0]
    i = 0
    j = num - 1
    c = False
    for i in range(num):
        pyi = polyy[i]
        pyj = polyy[j]
        pxi = polyx[i]
        pxj = polyx[j]
        if ((pyi > y) != (pyj > y)) and \
                (x < pxi + (pxj - pxi)*(y - pyi)/(pyj - pyi)):
            c = not c
        j = i
    return c
