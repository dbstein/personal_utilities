import numpy as np
import scipy as sp
import scipy.special
import scipy.interpolate

# seems like about 40 points is sufficient for machine accuracy of all sums
thetas, dtheta = np.linspace(0,2*np.pi,40,endpoint=False,retstep=True)
ct = np.cos(thetas)
st = np.sin(thetas)
ctst = ct*st
st2 = st*st
c2t = np.cos(2*thetas)
ct2 = ct**2
ct2c2t = ct2*c2t
ct4 = ct**4
ct3st = ct**3*st
myone = np.ones_like(thetas)

NN = 10000
mu = np.linspace(0.5,1.0,NN,endpoint=True)
mus = mu[:-1]
big_one = np.ones_like(mus)
newton_tol = 1e-12
jacobian_eps = 1e-6

def bingham_integral1(l1,f):
    Z = 2*np.pi*sp.special.iv(0,l1)
    integrand = np.exp(l1[:,None]*c2t[None,:])*f[None,:]
    return dtheta*np.sum(integrand, axis=-1)/Z

def full_bingham_integral(l1, B00, B01, B11, f):
    Z = 2*np.pi*sp.special.iv(0,l1)
    integrand = np.exp(B00[:,:,None]*ct2[None,None,:]+2*B01[:,:,None]*ctst[None,None,:]+B11[:,:,None]*st2[None,None,:])*f[None,None,:]
    return dtheta*np.sum(integrand, axis=-1)/Z

def get_jacobian(l1):
    I0 = sp.special.iv(0,l1)
    Im1 = sp.special.iv(-1,l1)
    Ip1 = sp.special.iv(1,l1)
    dI0 = (Im1+Ip1)/2.0
    Ja = -dI0/I0*bingham_integral1(l1,ct2)
    Jb = bingham_integral1(l1,ct2c2t)
    return Ja + Jb

def mu_to_lambda(mu):
    l1 = big_one*0.5
    err = bingham_integral1(l1,ct2) - mu
    err_max = np.abs(err).max()
    while err_max > newton_tol:
        jac = get_jacobian(l1)
        l1 -= err/jac
        err = bingham_integral1(l1,ct2) - mu
        err_max = np.abs(err).max()
    return l1

l1 = mu_to_lambda(mus)
# now integrate these to get S0000 and S0001 in the special coordinate system
S0000 = bingham_integral1(l1, ct4)
S0000 = np.concatenate((S0000, (1.0,)))
# get an interpolater for S0000
interper = sp.interpolate.interp1d(mu, S0000, kind='cubic', bounds_error=False, fill_value='extrapolate')

def rotate(l1, R):
    p = np.zeros_like(R)
    p[:,:,0,0] = l1
    p[:,:,1,1] = -l1
    return np.einsum('...ik,...kl,...jl->...ij',R,p,R)

def bingham_closure(D, E):
    """
    Direct Estimation of Bingham Closure (through rotation)
    """
    Dd = np.transpose(D, (2,3,0,1))
    EV = np.linalg.eigh(Dd)
    Eval = EV[0][:,:,::-1]
    Evec = EV[1][:,:,:,::-1]
    mu = Eval[:,:,0]
    mu[mu<0.5] = 0.5
    mu[mu>1.0] = 1.0
    tS0000 = interper(mu)
    tS0011 = Eval[:,:,0] - tS0000
    tS1111 = Eval[:,:,1] - tS0011
    # transform to real coordinates
    l00, l01, l10, l11 = Evec[:,:,0,0], Evec[:,:,0,1], Evec[:,:,1,0], Evec[:,:,1,1]
    S0000 = l00**4*tS0000 + 6*l01**2*l00**2*tS0011 + l01**4*tS1111
    S0001 = l00**3*l10*tS0000 + (3*l00*l01**2*l10+3*l00**2*l01*l11)*tS0011 + l01**3*l11*tS1111
    # get the others
    S0011 = D[0,0] - S0000
    S1111 = D[1,1] - S0011
    S0111 = D[0,1] - S0001
    # perform contractions
    SD = np.zeros_like(D)
    SD[0,0,:,:] = S0000*D[0,0] + 2*S0001*D[0,1] + S0011*D[1,1]
    SD[0,1,:,:] = S0001*D[0,0] + 2*S0011*D[0,1] + S0111*D[1,1]
    SD[1,1,:,:] = S0011*D[0,0] + 2*S0111*D[0,1] + S1111*D[1,1]
    SD[1,0,:,:] = SD[0,1]
    SE = np.zeros_like(E)
    SE[0,0,:,:] = S0000*E[0,0] + 2*S0001*E[0,1] + S0011*E[1,1]
    SE[0,1,:,:] = S0001*E[0,0] + 2*S0011*E[0,1] + S0111*E[1,1]
    SE[1,1,:,:] = S0011*E[0,0] + 2*S0111*E[0,1] + S1111*E[1,1]
    SE[1,0,:,:] = SE[0,1]
    return SD, SE
