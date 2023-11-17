import numpy as np
from near_finder.nufft_interp import periodic_interp1d

from .transformations import affine_transformation

def nufft_interpolation1d(x_out, in_hat, transformer=None):
	"""
	Interpolate fourier data given by in_hat to x_out locations
	x_out should either live in the range [0,2*pi] or an AffineTransformer
	should be specified that transforms it to that range
	"""
	if transformer is not None:
		x_out = transformer(x_out)
	interpolater = periodic_interp1d(fh=in_hat, eps=1e-14)
	return interpolater(x_out).real

	# out = np.zeros(x_out.shape[0], dtype=complex)
	# if old_nufft:
	# 	finufftpy.nufft1d2(x_out, out, 1, 1e-15, in_hat, modeord=1)
	# else:
	# 	finufft.nufft1d2(x_out, in_hat, out, isign=1, eps=1e-15, modeord=1)
	# adj = 1.0/in_hat.shape[0]
	# return out.real*adj
