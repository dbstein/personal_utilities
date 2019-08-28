import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg.lapack import zgetrs
from .chebyshev import get_chebyshev_nodes
from .fast_dot import fast_dot
from .single_liners import concat, reshape_to_vec
from .gmres import GmresSolver

default_gmres_type = 'right_scipy'

def fast_LU_solve(LU, b):
    """
    When running many small LU solves, the scipy call sp.linalg.lu_solve incurs
    significant overhead.  This calls the same LAPACK function, with no checks.

    Solves the system Ax=b for x, where LU = sp.linalg.lu_factor(A)
    (only for complex matrices and vectors...)
    """
    return zgetrs(LU[0], LU[1], b)[0]

def mfft(f):
    M = f.shape[0]
    N = f.shape[1]
    NS = N - 1
    N2 = int(N/2)
    fh = np.fft.fft(f)
    temp = np.empty((M, NS), dtype=complex)
    temp[:,:N2] = fh[:,:N2]
    temp[:,N2:] = fh[:,N2+1:]
    return temp
def mifft(fh):
    M = fh.shape[0]
    NS = fh.shape[1]
    N = NS + 1
    N2 = int(N/2)
    temp = np.empty((M, N), dtype=complex)
    temp[:,:N2]   = fh[:,:N2]
    temp[:,N2]    = 0.0
    temp[:,N2+1:] = fh[:,N2:]
    return np.fft.ifft(temp)
def fourier_multiply(fh, m):
    return mfft(m*mifft(fh))

def scalar_laplacian(CO, AAG, RAG, uh):
    R01 = CO.R01
    R12 = CO.R12
    D01 = CO.D01
    D12 = CO.D12
    ks = AAG.ks
    psi1 = RAG.psi1
    ipsi1 = RAG.inv_psi1
    ipsi2 = RAG.inv_psi2
    uh_t = R01.dot(uh*ks*1j)
    uh_tt = R12.dot(fourier_multiply(uh_t, ipsi1)*ks*1j)
    uh_rr = D12.dot(fourier_multiply(D01.dot(uh), psi1))
    luh = fourier_multiply(uh_rr+uh_tt, ipsi2)
    return luh

class ChebyshevOperators(object):
    def __init__(self, M, rat):
        self.M = M
        xc0, _ = np.polynomial.chebyshev.chebgauss(M-0)
        xc1, _ = np.polynomial.chebyshev.chebgauss(M-1)
        xc2, _ = np.polynomial.chebyshev.chebgauss(M-2)
        # vandermonde and inverse vandermonde matrices
        self.V0 = np.polynomial.chebyshev.chebvander(xc0, M-1)
        self.V1 = np.polynomial.chebyshev.chebvander(xc1, M-2)
        self.V2 = np.polynomial.chebyshev.chebvander(xc2, M-3)
        self.VI0 = np.linalg.inv(self.V0)
        self.VI1 = np.linalg.inv(self.V1)
        self.VI2 = np.linalg.inv(self.V2)
        # differentiation matrices
        DC01 = np.polynomial.chebyshev.chebder(np.eye(M-0)) / rat
        DC12 = np.polynomial.chebyshev.chebder(np.eye(M-1)) / rat
        DC00 = np.row_stack([DC01, np.zeros(M)])
        self.D00 = self.V0.dot(DC00.dot(self.VI0))
        self.D01 = self.V1.dot(DC01.dot(self.VI0))
        self.D12 = self.V2.dot(DC12.dot(self.VI1))
        # boundary condition operators
        self.ibc_dirichlet = np.polynomial.chebyshev.chebvander(1, M-1).dot(self.VI0)
        self.obc_dirichlet = np.polynomial.chebyshev.chebvander(-1, M-1).dot(self.VI0)
        self.ibc_neumann = self.ibc_dirichlet.dot(self.D00)
        self.obc_neumann = self.obc_dirichlet.dot(self.D00)
        # rank reduction operators
        temp = np.zeros([M-1, M-0], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.R01 = self.V1.dot(temp.dot(self.VI0))
        temp = np.zeros([M-2, M-1], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.R12 = self.V2.dot(temp.dot(self.VI1))
        self.R02 = self.R12.dot(self.R01)
        # get poof operator from M-1 --> M
        temp = np.zeros([M, M-1], dtype=float)
        np.fill_diagonal(temp, 1.0)
        self.P10 = self.V0.dot(temp.dot(self.VI1))

class ApproximateAnnularGeometry(object):
    """
    Approximate Annular Geometry for solving PDE in annular regions
    n: number of discrete points in tangential direction
    M: number of chebyshev modes in radial direction
    inner_radius: inner radius of annular region
    outer_radius: outer radius of annular region
    """
    def __init__(self, n, M, inner_radius, outer_radius):
        self.n = n
        self.M = M
        self.radius = outer_radius
        self.width = outer_radius - inner_radius
        self.radial_h = self.width/self.M
        self.tangent_h = 2*np.pi/n
        self.ns = self.n - 1
        self.n2 = int(self.n/2)
        self.k = np.fft.fftfreq(self.n, 1.0/self.n)
        self.ks = concat(self.k[:self.n2], self.k[self.n2+1:])
        # r grids
        _, self.rv0, rat0 = get_chebyshev_nodes(-self.width, 0.0, self.M-0)
        _, self.rv1, rat1 = get_chebyshev_nodes(-self.width, 0.0, self.M-1)
        _, self.rv2, rat2 = get_chebyshev_nodes(-self.width, 0.0, self.M-2)
        self.ratio = -rat0
        # coordinate transfromations
        self.approx_psi0 = self.radius+self.rv0
        self.approx_psi1 = self.radius+self.rv1
        self.approx_psi2 = self.radius+self.rv2
        self.approx_inv_psi0 = 1.0/self.approx_psi0
        self.approx_inv_psi1 = 1.0/self.approx_psi1
        self.approx_inv_psi2 = 1.0/self.approx_psi2
        # Chebyshev Operators
        self.CO = ChebyshevOperators(M, self.ratio)

class RealAnnularGeometry(object):
    def __init__(self, speed, curvature, AAG):
        k = np.fft.fftfreq(curvature.shape[0], 1.0/curvature.shape[0])
        dt_curvature = np.fft.ifft(np.fft.fft(curvature)*1j*k).real
        rv0 = AAG.rv0
        rv1 = AAG.rv1
        rv2 = AAG.rv2
        self.psi0 = speed*(1+rv0[:,None]*curvature)
        self.psi1 = speed*(1+rv1[:,None]*curvature)
        self.psi2 = speed*(1+rv2[:,None]*curvature)
        self.inv_psi0 = 1.0/self.psi0
        self.inv_psi1 = 1.0/self.psi1
        self.inv_psi2 = 1.0/self.psi2
        self.DR_psi2 = speed*curvature*np.ones(rv2[:,None].shape)
        denom2 = speed*(1+rv2[:,None]*curvature)**3
        idenom2 = 1.0/denom2
        # these are what i think it should be? need to check computation
        self.ipsi_DR_ipsi_DT_psi2 = (curvature-dt_curvature)*idenom2
        self.ipsi_DT_ipsi_DR_psi2 = -dt_curvature*idenom2
        # these are what work...
        self.ipsi_DR_ipsi_DT_psi2 = dt_curvature*idenom2
        self.ipsi_DT_ipsi_DR_psi2 = dt_curvature*idenom2

class WeirdAnnularModifiedHelmholtzSolver(object):
    def __init__(self, AAG, k, gmres_type=default_gmres_type):
        self.AAG = AAG
        self.k = k
        M =  AAG.M
        ns = AAG.ns
        n = AAG.n
        NB = M*ns
        self.M = M
        self.ns = ns
        self.n = n
        self.NB = NB
        self.small_shape = (self.M, self.ns)
        self.shape = (self.M, self.n)
        self._construct()
        self.gmres = GmresSolver(self._apply, self._preconditioner, complex, (NB, NB), gmres_type)
    def _construct(self):
        AAG = self.AAG
        CO = AAG.CO
        apsi1 =  AAG.approx_psi1
        aipsi1 = AAG.approx_inv_psi1
        aipsi2 = AAG.approx_inv_psi2
        ks =     AAG.ks
        D01 =    CO.D01
        D12 =    CO.D12
        R01 =    CO.R01
        R12 =    CO.R12
        R02 =    CO.R02
        ibcd =   CO.ibc_dirichlet
        ibcn =   CO.ibc_neumann
        ns =     self.ns
        M =      self.M
        self._KLUS = []
        for i in range(ns):
            K = np.empty((M,M), dtype=complex)
            LL = fast_dot(aipsi2, fast_dot(D12, fast_dot(apsi1, D01))) - \
                fast_dot(np.ones(M-2)*ks[i]**2, fast_dot(R12, fast_dot(aipsi1, R01)))
            K[:M-2] = self.k**2*R02 - LL
            K[M-2:M-1] = ibcd
            K[M-1:M-0] = ibcn
            self._KLUS.append(sp.linalg.lu_factor(K))
    def _preconditioner(self, fh):
        fh = fh.reshape(self.small_shape)
        fo = np.empty(self.small_shape, dtype=complex)
        for i in range(self.ns):
            fo[:,i] = fast_LU_solve(self._KLUS[i], fh[:,i])
        return fo.ravel()
    def _apply(self, uh):
        AAG = self.AAG
        RAG = self.RAG
        CO = self.AAG.CO
        ibcd = CO.ibc_dirichlet
        ibcn = CO.ibc_neumann
        R02  = CO.R02
        uh = uh.reshape(self.small_shape)
        luh = scalar_laplacian(CO, AAG, RAG, uh)
        fuh = self.k**2*R02.dot(uh) - luh
        ibc = ibcd.dot(uh)
        obc = ibcn.dot(uh)
        return concat(fuh, ibc, obc)
    def solve(self, RAG, f, idir, ineu, verbose=False, **kwargs):
        self.RAG = RAG
        R02 = self.AAG.CO.R02
        ff = concat(R02.dot(f), idir, ineu)
        ffh = mfft(ff.reshape(self.shape))
        res = self.gmres(ffh, **kwargs)
        if verbose:
            print('GMRES took:', self.gmres.iterations, 'iterations.')
        return mifft(res.reshape(self.small_shape)).real

class WeirdAnnularPoissonSolver(WeirdAnnularModifiedHelmholtzSolver):
    def __init__(self, AAG):
        super().__init__(AAG, 0.0)
    def solve(self, RAG, f, idir, ineu, verbose=False, **kwargs):
        return super().solve(RAG, -f, idir=idir, ineu=ineu, verbose=verbose, **kwargs)

class AnnularModifiedHelmholtzSolver(object):
    """
    Spectrally accurate Modified Helmholtz solver on annular domain

    Solves (k^2-L)u = f in the annulus described by the Annular Geometry AG
    Subject to the boundary condition:
    ia*u(ri) + ib*u_r(ri) = ig (boundary condition at the inner radius)
    oa*u(ro) + ob*u_r(ro) = og (boundary condition at the outer radius)

    On instantionation, a preconditioner is formed with ia, ib, ua, ub
        defining the boundary conditions
    These can be changed at solvetime, but preconditioning may not work so well
    """
    def __init__(self, AAG, k, ia=1.0, ib=0.0, oa=1.0, ob=0.0, gmres_type=default_gmres_type):
        self.AAG = AAG
        self.ia = ia
        self.ib = ib
        self.oa = oa
        self.ob = ob
        self.k = k
        M =  AAG.M
        ns = AAG.ns
        n = AAG.n
        NB = M*ns
        self.M = M
        self.ns = ns
        self.n = n
        self.NB = NB
        self.small_shape = (self.M, self.ns)
        self.shape = (self.M, self.n)
        self._construct()
        self.gmres = GmresSolver(self._apply, self._preconditioner, complex, (NB, NB), gmres_type)
    def _construct(self):
        AAG = self.AAG
        CO = AAG.CO
        apsi1 =  AAG.approx_psi1
        aipsi1 = AAG.approx_inv_psi1
        aipsi2 = AAG.approx_inv_psi2
        ks =     AAG.ks
        D01 =    CO.D01
        D12 =    CO.D12
        R01 =    CO.R01
        R12 =    CO.R12
        R02 =    CO.R02
        ibcd =   CO.ibc_dirichlet
        ibcn =   CO.ibc_neumann
        obcd =   CO.obc_dirichlet
        obcn =   CO.obc_neumann
        ns =     self.ns
        M =      self.M
        self._KLUS = []
        for i in range(ns):
            K = np.empty((M,M), dtype=complex)
            LL = fast_dot(aipsi2, fast_dot(D12, fast_dot(apsi1, D01))) - \
                fast_dot(np.ones(M-2)*ks[i]**2, fast_dot(R12, fast_dot(aipsi1, R01)))
            K[:M-2] = self.k**2*R02 - LL
            K[M-2:M-1] = self.ia*ibcd + self.ib*ibcn
            K[M-1:M-0] = self.oa*obcd + self.ob*obcn
            self._KLUS.append(sp.linalg.lu_factor(K))
    def _preconditioner(self, fh):
        fh = fh.reshape(self.small_shape)
        fo = np.empty(self.small_shape, dtype=complex)
        for i in range(self.ns):
            fo[:,i] = fast_LU_solve(self._KLUS[i], fh[:,i])
        return fo.ravel()
    def _apply(self, uh):
        AAG = self.AAG
        RAG = self.RAG
        CO = self.AAG.CO
        ibcd = CO.ibc_dirichlet
        ibcn = CO.ibc_neumann
        obcd = CO.obc_dirichlet
        obcn = CO.obc_neumann
        R02  = CO.R02
        uh = uh.reshape(self.small_shape)
        luh = scalar_laplacian(CO, AAG, RAG, uh)
        fuh = self.k**2*R02.dot(uh) - luh
        ibc = (self.ia*ibcd + self.ib*ibcn).dot(uh)
        obc = (self.oa*obcd + self.ob*obcn).dot(uh)
        return concat(fuh, ibc, obc)
    def solve(self, RAG, f, ig, og, ia=None, ib=None, oa=None, ob=None,
                                                    verbose=False, **kwargs):
        self.RAG = RAG
        self.ia = ia if ia is not None else self.ia
        self.ib = ib if ib is not None else self.ib
        self.oa = oa if oa is not None else self.oa
        self.ob = ob if ob is not None else self.ob
        R02 = self.AAG.CO.R02
        ff = concat(R02.dot(f), ig, og)
        ffh = mfft(ff.reshape(self.shape))
        res = self.gmres(ffh, **kwargs)
        if verbose:
            print('GMRES took:', self.gmres.iterations, 'iterations.')
        return mifft(res.reshape(self.small_shape)).real

class AnnularPoissonSolver(AnnularModifiedHelmholtzSolver):
    """
    Spectrally accurate Poisson solver on annular domain

    Solves Lu = f in the annulus described by the Annular Geometry AG
    Subject to the boundary condition:
    ia*u(ri) + ib*u_r(ri) = ig (boundary condition at the inner radius)
    oa*u(ro) + ob*u_r(ro) = og (boundary condition at the outer radius)

    On instantionation, a preconditioner is formed with ia, ib, ua, ub
        defining the boundary conditions
    These can be changed at solvetime, but preconditioning may not work so well
    """
    def __init__(self, AAG, ia=1.0, ib=0.0, oa=1.0, ob=0.0):
        super().__init__(AAG, 0.0, ia=ia, ib=ib, oa=oa, ob=ob)
    def solve(self, RAG, f, ig, og, ia=None, ib=None, oa=None, ob=None,
                                                    verbose=False, **kwargs):
        return super().solve(RAG, -f, ig=ig, og=og, ia=ia, ib=ib, oa=oa, ob=ob, \
                                                    verbose=verbose, **kwargs)

class AnnularStokesSolver(object):
    """
    Spectrally accurate Stokes solver on annular domain

    Solves -mu Lu + grad p = f in the annulus described by the Annular Geometry AG
                     div u = 0
    Subject to the boundary condition:
    u(ri) = ig (boundary condition at the inner radius)
    u(ro) = og (boundary condition at the outer radius)

    Note that this solver solves the problem in (r, t) coordinates
    And not in (u, v) coordinates

    The forces, and boundary conditions, must be given in this manner
    """
    def __init__(self, AAG, mu, gmres_type=default_gmres_type):
        self.AAG = AAG
        self.mu = mu
        M =  AAG.M
        ns = AAG.ns
        n = AAG.n
        self.M = M
        self.ns = ns
        self.n = n
        self.NU = self.M*self.ns
        self.NP = (self.M-1)*self.ns
        self.NB = 2*self.NU + self.NP
        self.u_small_shape = (self.M, self.ns)
        self.u_shape = (self.M, self.n)
        self.p_small_shape = (self.M-1, self.ns)
        self.p_shape = (self.M-1, self.n)
        self._construct()
        self.gmres = GmresSolver(self._apply, self._preconditioner, complex, (self.NB, self.NB), gmres_type)
    def _construct(self):
        AAG = self.AAG
        CO = AAG.CO
        apsi0 =  AAG.approx_psi0
        apsi1 =  AAG.approx_psi1
        aipsi1 = AAG.approx_inv_psi1
        aipsi2 = AAG.approx_inv_psi2
        ks =     AAG.ks
        D01 =    CO.D01
        D12 =    CO.D12
        R01 =    CO.R01
        R12 =    CO.R12
        R02 =    CO.R02
        ibcd =   CO.ibc_dirichlet
        obcd =   CO.obc_dirichlet
        VI1 =    CO.VI1
        ns =     self.ns
        M =      self.M
        self._KLUS = []
        for i in range(ns):
            K = np.zeros((3*M-1, 3*M-1), dtype=complex)
            LL = fast_dot(aipsi2, fast_dot(D12, fast_dot(apsi1, D01))) - \
                fast_dot(np.ones(M-2)*ks[i]**2, fast_dot(R12, fast_dot(aipsi1, R01)))
            # ur things
            K[ 0*M + 0   : 0*M + M-2  , 0*M : 1*M ] = -LL + fast_dot(aipsi2**2, R02)
            K[ 0*M + 0   : 0*M + M-2  , 1*M : 2*M ] = fast_dot(2*aipsi2**2, fast_dot(R02, 1j*ks[i]*np.ones(M)) )
            K[ 0*M + 0   : 0*M + M-2  , 0*M : 2*M ] *= self.mu
            K[ 0*M + 0   : 0*M + M-2  , 2*M :     ] = D12
            K[ 0*M + M-2 : 0*M + M-1  , 0*M : 1*M ] = ibcd
            K[ 0*M + M-1 : 0*M + M    , 0*M : 1*M ] = obcd
            # ut things
            K[ 1*M + 0   : 1*M + M-2  , 0*M : 1*M ] = -fast_dot(2*aipsi2**2, fast_dot(R02, 1j*ks[i]*np.ones(M)) )
            K[ 1*M + 0   : 1*M + M-2  , 1*M : 2*M ] = -LL + fast_dot(aipsi2**2, R02)
            K[ 1*M + 0   : 1*M + M-2  , 0*M : 2*M ] *= self.mu
            K[ 1*M + 0   : 1*M + M-2  , 2*M :     ] = fast_dot(aipsi2, fast_dot(R12, 1j*ks[i]*np.ones(M-1)))
            K[ 1*M + M-2 : 1*M + M-1  , 1*M : 2*M ] = ibcd
            K[ 1*M + M-1 : 1*M + M    , 1*M : 2*M ] = obcd
            # div u things
            K[ 2*M + 0   : 2*M + M-1  , 0*M : 1*M ] = fast_dot(aipsi1, fast_dot(D01, apsi0))
            K[ 2*M + 0   : 2*M + M-1  , 1*M : 2*M ] = fast_dot(aipsi1, fast_dot(R01, 1j*ks[i]*np.ones(M)))
            # fix the pressure nullspace
            if i == 0:
                K[2*M:, 2*M:] += VI1[0]
            self._KLUS.append(sp.linalg.lu_factor(K))
    def _preconditioner(self, ffh):
        M = self.M
        frh, fth, fph = self._extract_stokes(ffh, withcopy=True)
        for i in range(self.ns):
            vec = concat(frh[:,i], fth[:,i], fph[:,i])
            vec = fast_LU_solve(self._KLUS[i], vec)
            frh[:,i] = vec[0*M:1*M]
            fth[:,i] = vec[1*M:2*M]
            fph[:,i] = vec[2*M:]
        return concat(frh, fth, fph)
    def _extract_stokes(self, fh, withcopy=False):
        fh = reshape_to_vec(fh)
        frh = fh[0*self.NU:1*self.NU         ].reshape(self.u_small_shape)
        fth = fh[1*self.NU:2*self.NU         ].reshape(self.u_small_shape)
        fph = fh[2*self.NU:2*self.NU+self.NP ].reshape(self.p_small_shape)
        if withcopy:
            frh = frh.copy()
            fth = fth.copy()
            fph = fph.copy()
        return frh, fth, fph
    def _apply(self, uuh):
        AAG = self.AAG
        RAG = self.RAG
        CO = self.AAG.CO
        ibcd = CO.ibc_dirichlet
        obcd = CO.obc_dirichlet
        D01  = CO.D01
        D12  = CO.D12
        R01  = CO.R01
        R12  = CO.R12
        R02  = CO.R02
        VI1  = CO.VI1
        ks = AAG.ks
        psi0 = RAG.psi0
        psi1 = RAG.psi1
        ipsi1 = RAG.inv_psi1
        ipsi2 = RAG.inv_psi2
        DR_psi2 = RAG.DR_psi2
        ipsi_DR_ipsi_DT_psi2 = RAG.ipsi_DR_ipsi_DT_psi2
        ipsi_DT_ipsi_DR_psi2 = RAG.ipsi_DT_ipsi_DR_psi2
        # a lot of room for optimization in this function!
        uuh = reshape_to_vec(uuh)
        urh, uth, ph = self._extract_stokes(uuh)
        # compute scalar laplacian
        lap_urh = scalar_laplacian(CO, AAG, RAG, urh)
        lap_uth = scalar_laplacian(CO, AAG, RAG, uth)
        # ur equation
        t1 = fourier_multiply(R02.dot(uth*1j*ks), 2*DR_psi2*ipsi2**2)
        t2 = fourier_multiply(R02.dot(urh), DR_psi2**2*ipsi2**2)
        t3 = fourier_multiply(R02.dot(uth), ipsi_DR_ipsi_DT_psi2)
        t4 = D12.dot(ph)
        frh = self.mu*(-lap_urh + t1 + t2 + t3) + t4
        # ut equation
        t1 = fourier_multiply(R02.dot(urh*1j*ks), 2*DR_psi2*ipsi2**2)
        t2 = fourier_multiply(R02.dot(uth), DR_psi2**2*ipsi2**2)
        t3 = fourier_multiply(R02.dot(urh), ipsi_DT_ipsi_DR_psi2)
        t4 = fourier_multiply(R12.dot(ph*1j*ks), ipsi2)
        fth = self.mu*(-lap_uth - t1 + t2 - t3) + t4
        # div u equation
        fph = fourier_multiply(D01.dot(fourier_multiply(urh, psi0)) + R01.dot(uth*1j*ks), ipsi1)
        # add BCS
        frh_full = concat( frh, ibcd.dot(urh), obcd.dot(urh) )
        fth_full = concat( fth, ibcd.dot(uth), obcd.dot(uth) )
        # fph output
        fph[:,0] += VI1.dot(ph)[0,0] # this is the mean of the pressure!
        # get mean of the pressure
        return concat( frh_full, fth_full, fph )
    def solve(self, RAG, fr, ft, irg, itg, org, otg, verbose=False, **kwargs):
        self.RAG = RAG
        R02 = self.AAG.CO.R02
        P10 = self.AAG.CO.P10
        ffr = concat(R02.dot(fr), irg, org)
        fft = concat(R02.dot(ft), itg, otg)
        ffrh = mfft(ffr.reshape(self.u_shape))
        ffth = mfft(fft.reshape(self.u_shape))
        ffph = np.zeros(self.NP, dtype=complex)
        ffh = concat(ffrh, ffth, ffph)
        res = self.gmres(ffh, **kwargs)
        if verbose:
            print('GMRES took:', self.gmres.iterations, 'iterations.')
        urh, uth, ph = self._extract_stokes(res)
        ur = mifft(urh).real
        ut = mifft(uth).real
        p = P10.dot(mifft(ph).real)
        return ur, ut, p


