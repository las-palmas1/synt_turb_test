from generators.abstract import BCType, Generator, Block
from typing import List, Tuple
import numpy as np


class Lund(Generator):
    def __init__(self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float,
                 re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
                 re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray,
                 **kwargs):
        Generator.__init__(self, block, u_av, l_t, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, **kwargs)
        self._compute_aux_data()

    def _compute_cholesky(self):
        """Вычисление разложения тензора рейнольдсовых напряжений по Холецкому в каждой точке."""
        self.a11 = np.sqrt(self.re_xx)
        self.a12 = np.zeros(self.block.shape)
        self.a13 = np.zeros(self.block.shape)
        self.a21 = self.re_xy / self.a11
        self.a22 = np.sqrt(self.re_yy - self.a21**2)
        self.a23 = np.zeros(self.block.shape)
        self.a31 = self.re_xz / self.a11
        self.a32 = (self.re_yz - self.a21 * self.a31) / self.a22
        self.a33 = np.sqrt(self.re_zz - self.a31**2 - self.a32**2)

    def get_pulsation_at_node(self, i: int, j: int, k: int, time: np.ndarray) -> Tuple[float, float, float]:
        u_prime = np.random.normal(0, 1, time.shape)
        v_prime = np.random.normal(0, 1, time.shape)
        w_prime = np.random.normal(0, 1, time.shape)
        u = self.a11[i, j, k] * u_prime + self.a12[i, j, k] * v_prime + self.a13[i, j, k] * w_prime
        v = self.a21[i, j, k] * u_prime + self.a22[i, j, k] * v_prime + self.a23[i, j, k] * w_prime
        w = self.a31[i, j, k] * u_prime + self.a32[i, j, k] * v_prime + self.a33[i, j, k] * w_prime
        return u, v, w

    def get_velocity_field(self, time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_prime = np.random.normal(0, 1, self.block.shape)
        v_prime = np.random.normal(0, 1, self.block.shape)
        w_prime = np.random.normal(0, 1, self.block.shape)
        u = self.a11 * u_prime + self.a12 * v_prime + self.a13 * w_prime
        v = self.a21 * u_prime + self.a22 * v_prime + self.a23 * w_prime
        w = self.a31 * u_prime + self.a32 * v_prime + self.a33 * w_prime
        return u, v, w

    def _compute_aux_data(self):
        self._compute_cholesky()