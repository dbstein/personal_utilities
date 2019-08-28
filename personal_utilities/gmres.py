import numpy as np
import scipy as sp
import scipy.sparse.linalg
from .scipy_gmres import gmres, right_gmres
try:
    import pyamg
    pyamg_is_here = True
except:
    pyamg_is_here = False
try:
    from krypy.linsys import LinearSystem, Gmres
    krypy_is_here = True
except:
    krypy_is_here = False

LinearOperator = sp.sparse.linalg.LinearOperator

acceptable_solvers = ['scipy', 'modified_scipy', 'right_scipy']
if pyamg_is_here: acceptable_solvers += ['pyamg',]
if krypy_is_here: acceptable_solvers += ['krypy',]

class gmres_counter(object):
    def __init__(self):
        self.niter = 0
        self.residuals = []
    def __call__(self, rk=None):
        self.niter += 1
        self.residuals.append(np.abs(rk).max())

class GmresSolver(object):
    def __init__(self, A_func, M_func, dtype, shape, solver):
        """
        Setup a GMRES solver for the system Ax=..., where A_func applies A
        M_func applies an approximate inverse to A
        solver is the solver used, and can be one of:
            'pyamg'
            'krypy'
            'scipy'
            'modified_scipy'
        """
        self.shape = shape
        self.A = LinearOperator(shape, dtype=dtype, matvec=A_func)
        self.M = LinearOperator(shape, dtype=dtype, matvec=M_func)
        self._set_solver(solver)
    def _set_solver(self, solver):
        assert solver in acceptable_solvers, 'Requested solver not known'
        self.solver = solver
    def __call__(self, b, tol=1e-8, maxiter=100, restart=100):
        if self.solver == 'krypy':
            self._solve_krypy(b, tol, maxiter)
        elif self.solver == 'pyamg':
            self._solve_pyamg(b, tol, maxiter, restart)
        elif self.solver == 'scipy':
            self._solve_scipy(b, tol, maxiter, restart)
        elif self.solver == 'modified_scipy':
            self._solve_modified_scipy(b, tol, maxiter, restart)
        elif self.solver == 'right_scipy':
            self._solve_right_scipy(b, tol, maxiter, restart)
        else:
            raise Exception('Gmres solver not found.')
        return self.result
    def _solve_krypy(self, b, tol, maxiter):
        linear_system = LinearSystem(self.A, b.ravel(), Ml=self.M)
        self.output = Gmres(linear_system, maxiter=maxiter, tol=tol)
        self.iterations = len(self.output.resnorms)
        self.result = self.output.xk.reshape(self.shape[0])
    def _solve_pyamg(self, b, tol, maxiter, restart):
        counter = gmres_counter()
        self.output = pyamg.krylov.gmres(self.A, b.ravel(), M=self.M, tol=tol, \
                              maxiter=maxiter, restrt=restart, callback=counter)
        self.counter = counter
        self.iterations = counter.niter
        self.result = self.output[0].reshape(self.shape[0])
    def _solve_scipy(self, b, tol, maxiter, restart):
        counter = gmres_counter()
        self.output = sp.sparse.linalg.gmres(self.A, b.ravel(), M=self.M, tol=tol, \
                              maxiter=maxiter, restart=restart, callback=counter)
        self.counter = counter
        self.iterations = counter.niter
        self.result = self.output[0].reshape(self.shape[0])
    def _solve_modified_scipy(self, b, tol, maxiter, restart):
        self.output = gmres(self.A, b.ravel(), tol=tol, restart=restart, maxiter=maxiter, M=self.M, verbose=False)
        self.result = self.output[0].reshape(self.shape[0])
        self.iterations = len(self.output[2])
    def _solve_right_scipy(self, b, tol, maxiter, restart):
        self.output = right_gmres(self.A, b.ravel(), tol=tol, restart=restart, maxiter=maxiter, M=self.M, verbose=False)
        self.result = self.output[0].reshape(self.shape[0])
        self.iterations = len(self.output[2])



