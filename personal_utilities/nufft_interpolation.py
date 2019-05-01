import numpy as np
import finufftpy

from .transformations import affine_transformation

def nufft_interpolation1d(x_out, in_hat, transformer=None):
	"""
	Interpolate fourier data given by in_hat to x_out locations
	x_out should either live in the range [0,2*pi] or an AffineTransformer
	should be specified that transforms it to that range
	"""
	if transformer is not None:
		x_out = transformer(x_out)
	out = np.zeros(x_out.shape[0], dtype=complex)
	finufftpy.nufft1d2(x_out, out, 1, 1e-15, in_hat, modeord=1)
	adj = 1.0/in_hat.shape[0]
	return out.real*adj
