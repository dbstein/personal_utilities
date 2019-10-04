import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as sspl
from personal_utilities.scipy_gmres import gmres, right_gmres

"""
Demonstrate how to use the scipy_gmres interfaces to solve some linear systems
"""

def get_err(ref, sol):
	return np.abs(ref-sol).max()

##### First a simple, well-conditioned problem
print('Well-conditioned problem')
n = 1000

A = np.eye(n) + 0.1*np.random.rand(n, n)
b = np.random.rand(n)
x0 = np.linalg.solve(A, b)
# non-preconditioned gmres
x1 = gmres(A, b, verbose=False, tol=1e-14)
print('    Not-preconditioned --          Error: {:0.2e}'.format(get_err(x0, x1[0])), 'Iterations:', len(x1[2]))
# preconditioned gmres
AILU = sspl.spilu(ssp.csc_matrix(A), drop_tol=1e-4)
prec = sspl.LinearOperator(shape=A.shape, dtype=A.dtype, matvec=AILU.solve)
x1 = gmres(A, b, verbose=False, tol=1e-14, M=prec)
print('    Left-preconditioned, resid --  Error: {:0.2e}'.format(get_err(x0, x1[0])), 'Iterations:', len(x1[2]))
x1 = gmres(A, b, verbose=False, tol=1e-14, M=prec, convergence='presid')
print('    Left-preconditioned, presid -- Error: {:0.2e}'.format(get_err(x0, x1[0])), 'Iterations:', len(x1[2]))
x1 = right_gmres(A, b, verbose=False, tol=1e-14, M=prec)
print('    Right-preconditioned        -- Error: {:0.2e}'.format(get_err(x0, x1[0])), 'Iterations:', len(x1[2]))

##### Now an ill-conditioned example (this is not a good example...)
print('Ill-conditioned problem')
n = 1000
x, h = np.linspace(0, 1, n+1, retstep=True)
f = (12*x**2 + 12*x)*(1+x)
u = x**4 + 2*x**3 + 2*x
u0 = 0
u1 = 5
L = ssp.lil_matrix(np.eye(n-1)*(-2))
L.setdiag(1.0, 1)
L.setdiag(1.0, -1)
L /= h**2
A = np.array(L.todense())
A *= (1+x[1:-1][:,None])
B = np.array(L.todense())
BI = np.linalg.inv(B)

b = f[1:-1].copy()
b[0]  -= u0/h**2*(1+x[1])
b[-1] -= u1/h**2*(1+x[-2])

tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

for tol in tols:
	print('    Tolerance is: {:0.1e}'.format(tol))
	x1 = gmres(A, b, verbose=False, tol=tol, M=BI, maxiter=1000)
	print('        Left-preconditioned, resid --  Error: {:0.2e}'.format(get_err(u[1:-1], x1[0])), 'Iterations:', len(x1[2]))
	x1 = gmres(A, b, verbose=False, tol=tol, M=BI, convergence='presid')
	print('        Left-preconditioned, presid -- Error: {:0.2e}'.format(get_err(u[1:-1], x1[0])), 'Iterations:', len(x1[2]))
	x1 = right_gmres(A, b, verbose=False, tol=tol, M=BI)
	print('        Right-preconditioned        -- Error: {:0.2e}'.format(get_err(u[1:-1], x1[0])), 'Iterations:', len(x1[2]))

