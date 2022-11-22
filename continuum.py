## computing the continuum
import numpy as np
from numpy.linalg.linalg import eig

nax = np.newaxis

class Continuum:
    def __init__(self, island, mstart=0, mcount=5, nstart=1, ncount=5, mfp=1, nfp=1, fft_multiplier=4):
        self.island = island
        self.mstart = mstart
        self.mcount = mcount
        self.nstart = nstart
        self.ncount = ncount
        self.nfp = nfp
        self.fft_multiplier = fft_multiplier

        self.mlist = np.arange(mstart, mstart + mfp * mcount, mfp)
        self.nlist = np.arange(nstart, nstart + nfp * ncount, nfp)

        self.nt = self.mcount * fft_multiplier
        self.nz = self.ncount * fft_multiplier

        self.tgrid = np.linspace(0, 2*np.pi / mfp, self.nt, endpoint=False)
        self.zgridp = np.linspace(0, 2*np.pi / nfp, self.nz, endpoint=False)
        self.zgridt = np.linspace(0, 2*np.pi / nfp * island.m0, self.nz, endpoint=False)

    def assemble_matrix_passing(self, chi, sign=1):
        psibar, omega, q = self.island.get_psibar_omega_q_passing(chi, sign=sign)
        J, gradchi2, B2 = self.island.get_J_gradchi2_B2(chi, self.tgrid, self.zgridp, passing=True, sign=sign)

        I = gradchi2 * J / B2
        W = I / J**2

        Ifft = np.fft.fft2(I) / self.nt / self.nz
        Wfft = np.fft.fft2(W) / self.nt / self.nz

        mlist = np.broadcast_to(self.mlist[:, nax], [self.mcount, self.ncount]).flatten()
        nlist = np.broadcast_to(self.nlist[nax, :], [self.mcount, self.ncount]).flatten()

        mqn = mlist / q + nlist
        mqnmat = mqn[:, nax] * mqn[nax, :]

        midlist = np.arange(0,self.mcount)
        midlist = np.broadcast_to(midlist[:, nax], [self.mcount, self.ncount]).flatten()
        nidlist = np.arange(0,self.ncount)
        nidlist = np.broadcast_to(nidlist[nax, :], [self.mcount, self.ncount]).flatten()

        diffm = midlist[:, nax] - midlist[nax, :]
        diffn = nidlist[:, nax] - nidlist[nax, :]

        Imat = Ifft[diffm, diffn]
        Wmat = Wfft[diffm, diffn] * mqnmat

        return Imat, Wmat

    def assemble_matrix_trapped(self, chi):
        psibar, omega, q = self.island.get_psibar_omega_q_trapped(chi)
        J, gradchi2, B2 = self.island.get_J_gradchi2_B2(chi, self.tgrid, self.zgridt, passing=False)

        I = gradchi2 * J / B2
        W = I / J**2

        Ifft = np.fft.fft2(I) / self.nt / self.nz
        Wfft = np.fft.fft2(W) / self.nt / self.nz

        mlist = np.broadcast_to(self.mlist[:, nax], [self.mcount, self.ncount]).flatten()
        nlist = np.broadcast_to(self.nlist[nax, :], [self.mcount, self.ncount]).flatten()

        mqn = mlist / q + nlist / self.island.m0
        mqnmat = mqn[:, nax] * mqn[nax, :]

        midlist = np.arange(0,self.mcount)
        midlist = np.broadcast_to(midlist[:, nax], [self.mcount, self.ncount]).flatten()
        nidlist = np.arange(0,self.ncount)
        nidlist = np.broadcast_to(nidlist[nax, :], [self.mcount, self.ncount]).flatten()

        diffm = midlist[:, nax] - midlist[nax, :]
        diffn = nidlist[:, nax] - nidlist[nax, :]

        Imat = Ifft[diffm, diffn]
        Wmat = Wfft[diffm, diffn] * mqnmat

        return Imat, Wmat

    def compute_continuum(self, chilist, passing=True, sign=1, eigenvector=False, **kwargs):
        import scipy.linalg as linalg   
        omega2list = [];
        vlist = [];
        if passing:
            psibar, _, _ = self.island.get_psibar_omega_q_passing(chilist, sign=sign)
        else:
            psibar, _, _ = self.island.get_psibar_omega_q_trapped(chilist)

        for chi in chilist:
            if passing:
                Imat, Wmat = self.assemble_matrix_passing(chi, sign)
            else:
                Imat, Wmat = self.assemble_matrix_trapped(chi)

            w, v = linalg.eigh(
                a=Wmat,
                b=Imat,
                **kwargs
            )

            omega2list.append(w)
            if eigenvector:
                vlist.append(v)

        if eigenvector:
            return psibar, np.array(omega2list), vlist
        else:
            return psibar, np.array(omega2list)




