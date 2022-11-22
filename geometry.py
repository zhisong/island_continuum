## specifying the geometry including gij and jacobian
import numpy as np

class Geometry:
    """! This is the basic geometry class assuming a slab
    """
    def __init__(self, a=1, r0=1, R0=3, B0=1):
        self.a = a
        self.r0 = r0
        self.R0 = R0
        self.B0 = B0

    def get_metric(self, psi, theta, zeta):

        g11 = np.ones_like(psi) * self.B0**2 * self.r0**2
        g12 = np.zeros_like(psi)
        g22 = np.ones_like(psi) / self.r0**2
        g33 = np.ones_like(psi) / self.R0**2

        J =  np.ones_like(psi) / (self.B0 / self.R0)

        return g11, g12, g22, g33, J


class ToroidalGeometry:
    """! This is the geometry class for a toroidal system
    """
    def __init__(self, a=1, R0=3, B0=1):
        self.a = a
        self.R0 = R0
        self.B0 = B0

    def get_metric(self, psi, theta, zeta):

        R0 = self.R0
        B0 = self.B0
        r = np.sqrt(2 * psi / B0)
        dpsidr = r * B0
        deltaprime = r / R0 / 4
        ddeltaprime = 1 / R0 / 4
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        J = r * R0 * (1 + 2 * r / R0 * costheta) / dpsidr

        g11 = dpsidr ** 2 * (1 + 2 * deltaprime * costheta)
        g12 = -dpsidr / r * sintheta * (r / R0 + deltaprime + r * ddeltaprime)
        g22 = 1 / r**2 * (1 - 2*(r/R0 + deltaprime) * costheta)
        g33 = 1 / self.R0**2 / (1 + 2 * r / R0 * costheta)

        return g11, g12, g22, g33, J

    def get_RZ(self, psi, theta, zeta):
        R0 = self.R0
        B0 = self.B0
        r = np.sqrt(2 * psi / B0)
        dpsidr = r * B0
        delta = r**2/R0 / 8
        deltaprime = r / R0 / 4
        ddeltaprime = 1 / R0 / 4
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        eta = (r/R0 + deltaprime)/2
        
        R = R0 + r * costheta  - delta + r * eta * (np.cos(2*theta) - 1)
        Z = r * sintheta + r * eta(r) * np.sin(2 * theta),