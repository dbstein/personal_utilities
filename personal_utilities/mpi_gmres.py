from scipy.sparse.linalg.isolve.utils import make_system
from scipy.sparse.linalg.isolve import _iterative
from scipy._lib._util import _aligned_zeros
import numpy as np
import scipy
from functools import partial

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        mydict['resnorms'] = []
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
        try:
            mydict['resnorms'].append(rk)
        except:
            pass
mydict = {
}

def _stoptest(residual, atol):
    resid = np.linalg.norm(residual)
    if resid <= atol:
        return resid, 1
    else:
        return resid, 0
_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}

w1 = scipy.__version__.split('.')
scipy_version = int(w1[0]) + int(w1[1])/10.0 + int(w1[2])/100.0

class null_linop(object):
    def __init__(self):
        pass
    def __call__(self, x):
        return x
null_preconditioner = null_linop()

class mpi_linop(object):
    def __init__(self, matvec, n, dtype, comm, rank):
        self.matvec = matvec
        self.n = n
        self.dtype = dtype
        self.comm = comm
        self.rank = rank
    def __call__(self, x):
        return self.matvec(x, self.comm, self.rank)

def presid_gmres(A, M, b, verbose, x0=None, tol=1e-05, restart=None, maxiter=None):
    matvec = A
    psolve = M
    comm = A.comm
    rank = A.rank
    n = A.n
    dtype = A.dtype
    isMaster = rank == 0

    callback = gmres_counter(verbose) if isMaster else None
    x = np.zeros(n, dtype=dtype) if x0 is None else x0

    if maxiter is None: maxiter = n*10
    if restart is None: restart = 20
    restart = min(restart, n)

    ltr = _type_conv[np.dtype(dtype).char]
    revcom = getattr(_iterative, ltr + 'gmresrevcom')
    
    pb = psolve(b)
    mb = matvec(b)

    if rank == 0:
        bnrm2 = np.linalg.norm(b)
        Mb_nrm2 = np.linalg.norm(pb)
        get_residual = lambda: np.linalg.norm(mb - b)
        atol = tol
    else:
        bnrm2 = None
    bnrm2 = comm.bcast(bnrm2, root=0)

    if bnrm2 == 0:
        return postprocess(b), 0

    if rank == 0:
        # Tolerance passed to GMRESREVCOM applies to the inner iteration
        # and deals with the left-preconditioned residual.
        ptol_max_factor = 1.0
        ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
        resid = np.nan
        presid = np.nan
        # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
        work2 = _aligned_zeros((restart+1)*(2*restart+2),dtype=dtype)
        info = 0
        ftflag = True
        iter_ = maxiter
        first_pass = True
        resid_ready = False
    ndx1 = 1
    ndx2 = -1
    ijob = 1
    old_ijob = ijob
    work = _aligned_zeros((6+restart)*n,dtype=dtype)
    iter_num = 1

    while True:
        if rank == 0:
            ### begin my modifications
            if presid/bnrm2 < atol:
                resid = presid/bnrm2
                info = 1
            if info: ptol = 10000
            ### end my modifications
            x, iter_, presid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
               revcom(b, x, restart, work, work2, iter_, presid, info, ndx1, ndx2, ijob, ptol)
        ijob = comm.bcast(ijob, root=0)
        ndx1 = comm.bcast(ndx1, root=0)
        ndx2 = comm.bcast(ndx2, root=0)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):  # gmres success, update last residual
            if rank == 0:
                if resid_ready and callback is not None:
                    callback(presid / bnrm2)
                    resid_ready = False
            break
        elif (ijob == 1):
            upd = matvec(x)
            if rank == 0:
                work[slice2] *= sclr2
                work[slice2] += sclr1*upd
        elif (ijob == 2):
            upd = psolve(work[slice2])
            if rank == 0:
                work[slice1] = upd
                if not first_pass and old_ijob == 3:
                    resid_ready = True
                first_pass = False
        elif (ijob == 3):
            upd = matvec(work[slice1])
            if rank == 0:
                work[slice2] *= sclr2
                work[slice2] += sclr1*upd
                if resid_ready and callback is not None:
                    callback(presid / bnrm2)
                    resid_ready = False
                    iter_num = iter_num+1
        elif (ijob == 4):
            if rank == 0:
                if ftflag:
                    info = -1
                    ftflag = False
                resid, info = _stoptest(work[slice1], atol)
                # Inner loop tolerance control
                if info or presid > ptol:
                    ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
                else:
                    # Inner loop tolerance OK, but outer loop not.
                    ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

                if resid != 0:
                    ptol = presid * min(ptol_max_factor, atol / resid)
                else:
                    ptol = presid * ptol_max_factor

        if rank == 0:
            old_ijob = ijob
            ijob = 2

        # need to set ijob according to the master rank
        ijob = comm.bcast(ijob, root=0)
        iter_num = comm.bcast(iter_num, root=0)

        if iter_num > maxiter:
            info = maxiter
            break

    if rank == 0:
        if info >= 0 and not (resid <= atol):
            # info isn't set appropriately otherwise
            info = maxiter
    
        return x, info, mydict['resnorms']
    else:
        return None

def gmres(A, b, M=null_preconditioner, verbose=False, **kwargs):
    """
    MPI based GMRES
    """
    return presid_gmres(A, M, b, verbose, **kwargs)

def right_gmres(A, b, M, verbose=False, **kwargs):
    """
    (thanks to Floren Balboa-Usabiaga for the code)
    """
    def APinv(x, comm, rank):
        return A(M(x))
    AO = mpi_linop(APinv, A.n, A.dtype, A.comm, A.rank)

    # Solve system A*P^{-1} * y = b
    out = presid_gmres(AO, null_preconditioner, b, verbose, **kwargs)
    if A.rank == 0:
        y, info, resnorms = out
    else:
        y = None

    # Solve system P*x = y
    x = M(y)

    if A.rank == 0:
        # Return solution and info
        return x, info, resnorms
    else:
        return None

