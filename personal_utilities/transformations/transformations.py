import numpy as np
import numexpr as ne

def affine_transformation(xin, min_in, max_in, min_out, max_out, 
                            return_ratio=False, use_numexpr=False, xout=None):
    ran_in = max_in - min_in
    ran_out = max_out - min_out
    rat = ran_out/ran_in
    xout = _affine_transformation(xin, rat, min_in, min_out, use_numexpr)
    if return_ratio:
        out = xout, rat
    else:
        out = xout
    return out

def _affine_transformation(xin, rat, min_in, min_out, use_numexpr):
    if use_numexpr:
        xout = ne.evaluate('(x-min_in)*rat + min_out', out=xout)
    else:
        xout = (x - min_in)*rat + min_out
    return xout

class AffineTransformer(object):
    def __init__(self, min_in, max_in, min_out, max_out, use_numexpr=False):
        self.min_in  = min_in
        self.max_in  = max_in
        self.min_out = min_out
        self.max_out = max_out
        self.ran_in  = max_in - min_in
        self.ran_out = max_out - min_out
        self.ratio   = ran_out/ran_in
        self.use_numexpr = use_numexpr
    def __call__(self, xin):
        return _affine_transformation(xin, self.ratio, self.min_in,
                                            self.min_out, self.use_numexpr)
    def get_ratio(self):
        return self.ratio
