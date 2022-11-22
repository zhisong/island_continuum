## specifying the island
import numpy as np
from numpy.core.numeric import zeros_like
from scipy.special import ellipk, ellipe, ellipj

nax = np.newaxis


class Island:
    """! This is specifiying a local treatment for an island"""

    def __init__(self, geometry, m0=5, n0=2, psi0=0.1, qprime=1, A=0.1):
        self.geometry = geometry
        self.m0 = m0
        self.n0 = n0
        self.q0 = m0 / n0
        self.qprime = qprime
        self.psi0 = psi0
        self.A = A

    def get_psibar_omega_q_passing(self, chi, sign=1):
        kappa = (self.A - chi) / (2 * self.A)
        w = 4 * np.sqrt(self.A * self.q0 ** 2 / self.qprime)

        omega = (
            -sign
            * np.sqrt(self.A * self.qprime / self.q0 ** 2)
            * np.pi
            * np.sqrt(kappa)
            / ellipk(1 / kappa)
        )
        psibar = self.psi0 + sign * w / np.pi * np.sqrt(kappa) * ellipe(1 / kappa)

        q = 1 / (omega + 1 / self.q0)

        return psibar, omega, q

    def get_psibar_omega_q_trapped(self, chi):
        # chi = 0.6037057617728531 * self.A
        kappa = (self.A - chi) / (2 * self.A)
        w = 4 * np.sqrt(self.A * self.q0 ** 2 / self.qprime)

        omega = (
            -np.sqrt(self.A * self.qprime / self.q0 ** 2)
            * self.m0
            * np.pi
            / 2
            / ellipk(kappa)
        )
        psibar = 2 * w / self.m0 / np.pi * ((kappa - 1) * ellipk(kappa) + ellipe(kappa))

        q = 1 / (omega + 1 / self.q0 * 0)

        return psibar, omega, q

    def get_psi_and_theta_passing(self, chi, thetabar, zeta, sign=1):

        kappa = self._chi_to_kappa(chi)

        alphabar = thetabar[:, nax] - zeta[nax, :] / self.q0
        alpha = self._alpha_passing(alphabar, kappa)
        theta = alpha + zeta[nax, :] / self.q0

        psi = (
            np.sqrt(2 * self.q0 ** 2 / self.qprime)
            * sign
            * np.sqrt(self.A * np.cos(self.m0 * alpha) - chi)
            + self.psi0
        )

        return psi, theta

    def get_psi_and_theta_trapped(self, chi, thetabar, zeta):
        # chi = 0.6037057617728531 * self.A
        kappa = self._chi_to_kappa(chi)

        alphabar = thetabar[:, nax] - zeta[nax, :] / self.q0 * 0
        alpha = self._alpha_trapped(alphabar, kappa)
        theta = alpha + zeta[nax, :] / self.q0

        psi = np.sqrt(2 * self.q0 ** 2 / self.qprime) * np.sqrt(
            np.abs(self.A * np.cos(self.m0 * alpha) - chi)
        )

        sign = 2 * np.mod((alphabar - np.pi / 2) // (np.pi), 2) - 1

        psi = psi * sign + self.psi0

        return psi, theta

    def get_J_gradchi2_B2(self, chi, thetabar, zeta, passing=True, sign=1):
        if passing:
            psi, theta = self.get_psi_and_theta_passing(chi, thetabar, zeta, sign)
        else:
            psi, theta = self.get_psi_and_theta_trapped(chi, thetabar, zeta)

        zeta = np.broadcast_to(zeta[nax, :], psi.shape)
        alpha = theta - zeta / self.q0

        g11, g12, g22, g33, J = self.geometry.get_metric(psi, theta, zeta)

        gradchi2 = (
            self.qprime ** 2 / self.q0 ** 4 * (psi - self.psi0) ** 2 * g11
            + self.A ** 2
            * self.m0 ** 2
            * np.sin(self.m0 * alpha) ** 2
            * (g22 + g33 / self.q0 ** 2)
            + 2
            * self.qprime
            / self.q0 ** 2
            * (psi - self.psi0)
            * self.A
            * self.m0
            * np.sin(self.m0 * alpha)
            * g12
        )

        g00 = zeros_like(g11)

        gupper = np.array([[g11, g12, g00], [g12, g22, g00], [g00, g00, g33]])
        gupper = np.moveaxis(gupper, (0, 1), (-2, -1))
        glower = np.linalg.inv(gupper)

        q = 1 / (1 / self.q0 - self.qprime / self.q0 ** 2 * (psi - self.psi0))

        Bz = 1 / J
        Bt = 1 / J / q
        Bpsi = self.A * self.m0 * np.sin(self.m0 * alpha) / J

        Bupper = np.stack([Bpsi, Bt, Bz], -1)

        B2 = np.einsum("...i,...ij,...j->...", Bupper, glower, Bupper)

        # B = 1 - 0.17197397050759053 * np.cos(zeta / self.q0)
        # B2 = B ** 2
        # J = 1 / B2 * self.geometry.R0
        # x = chi
        # print(x)
        # gradchi2 = (
        #     1
        #     + 0.8818 * np.cos(np.broadcast_to(thetabar[:, nax], psi.shape) * 2)
        #     + 0.11031105375332835 * x * np.cos(2 * thetabar[:, nax] + zeta / self.q0)
        #     - 0.007303335450597716 * np.cos(2 * thetabar[:, nax] - zeta / self.q0)
        #     + 0.05381156055800194 * np.cos(zeta / self.q0)
        # )

        return J, gradchi2, B2

    def _alpha_passing(self, alphabar, kappa):
        K = ellipk(1 / kappa)
        sn, cn, dn, am = ellipj(self.m0 * K / np.pi * alphabar, 1 / kappa)
        alpha = 2 / self.m0 * am

        return alpha

    def _alpha_trapped(self, alphabar, kappa):
        K = ellipk(kappa)
        sn, cn, dn, am = ellipj(2 * K / np.pi * alphabar, kappa)
        alpha = 2 / self.m0 * np.arcsin(np.sqrt(kappa) * sn)

        return alpha

    def _chi_to_kappa(self, chi):
        return (self.A - chi) / (2 * self.A)