import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as sspl
from personal_utilities.mpi_gmres import gmres, right_gmres
from personal_utilities.mpi_gmres import mpi_linop

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Demonstrate how to use the gmres, right_gmres functions in mpi_gmres
(this should be run with mpirun -2 4 python mpi_gmres_usage.py)

Note that all the "GMRES" stuff is done on a single rank (that is, rank 0)
But the linear operator and preconditioner may be parallelized...
"""

def get_err(ref, sol):
	return np.abs(ref-sol).max()

# Simple example with a block-diagonal problem
n = 100

A = np.eye(n) + 0.1*np.random.rand(n, n)
b = np.random.rand(n)
s = np.linalg.solve(A, b)

full_B = comm.gather(b)
full_S = comm.gather(s)

if rank  == 0:
	full_b = np.concatenate(full_B)
	full_s = np.concatenate(full_S)
else:
	full_b = None

def full_A_func(x, comm, rank):
	xx = [x[:n], x[n:]] if rank == 0 else None
	w = comm.scatter(xx, root=0)
	o = A.dot(w)
	oo = comm.gather(o)
	return np.concatenate(oo) if rank == 0 else None
full_A = mpi_linop(full_A_func, 2*n, float, comm, rank)

# non-preconditioned gmres
if rank == 0: print('\nNo preconditioning')
out = gmres(full_A, full_b, verbose=True, tol=1e-14)
if rank == 0:
	full_e1 = out[0]
	print('Error is: {:0.2e}'.format(get_err(full_s, full_e1)))

# ILU preconditioner
AILU = sspl.spilu(ssp.csc_matrix(A), drop_tol=1e-3)
def full_P_func(x, comm, rank):
	xx = [x[:n], x[n:]] if rank == 0 else None
	w = comm.scatter(xx, root=0)
	o = AILU.solve(w)
	oo = comm.gather(o)
	return np.concatenate(oo) if rank == 0 else None
full_P = mpi_linop(full_P_func, 2*n, float, comm, rank)

# preconditioned gmres
if rank == 0: print('\nILU preconditioning, left')
out = gmres(full_A, full_b, full_P, verbose=True, tol=1e-14)
if rank == 0:
	full_e2 = out[0]
	print('Error is: {:0.2e}'.format(get_err(full_s, full_e2)))

# right-preconditioned gmres
if rank == 0: print('\nILU preconditioning, right')
out = right_gmres(full_A, full_b, full_P, verbose=True, tol=1e-14)
if rank == 0:
	full_e3 = out[0]
	print('Error is: {:0.2e}'.format(get_err(full_s, full_e3)))


