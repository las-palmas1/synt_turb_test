from generators.abstract import BCType, Generator, Block
from typing import List, Tuple
import numpy as np
import numpy.linalg as la
from generators.tools import smirnov_compute_velocity_field


class Lund(Generator):
    def __init__(self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float,
                 re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
                 re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray):
        Generator.__init__(self, block, u_av, l_t, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz)
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


class Smirnov(Generator):
    def __init__(self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float, tau_t: float,
                 re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
                 re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray,
                 mode_num: int=100,):
        Generator.__init__(self, block, u_av, l_t, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz)
        self.tau_t = tau_t
        self.mode_num = mode_num
        self._compute_aux_data()

    def _compute_eigen_values(self):
        self.c1 = np.zeros(self.block.shape)
        self.c2 = np.zeros(self.block.shape)
        self.c3 = np.zeros(self.block.shape)
        self.a11 = np.zeros(self.block.shape)
        self.a12 = np.zeros(self.block.shape)
        self.a13 = np.zeros(self.block.shape)
        self.a21 = np.zeros(self.block.shape)
        self.a22 = np.zeros(self.block.shape)
        self.a23 = np.zeros(self.block.shape)
        self.a31 = np.zeros(self.block.shape)
        self.a32 = np.zeros(self.block.shape)
        self.a33 = np.zeros(self.block.shape)
        for i in range(self.block.shape[0]):
            for j in range(self.block.shape[1]):
                for k in range(self.block.shape[2]):
                    mat = np.array([
                        [self.re_xx[i, j, k], self.re_xy[i, j, k], self.re_xz[i, j, k]],
                        [self.re_xy[i, j, k], self.re_yy[i, j, k], self.re_yz[i, j, k]],
                        [self.re_xz[i, j, k], self.re_yz[i, j, k], self.re_zz[i, j, k]],
                    ])
                    w, v = la.eig(mat)
                    self.c1[i, j, k] = np.sqrt(w[0])
                    self.c2[i, j, k] = np.sqrt(w[1])
                    self.c3[i, j, k] = np.sqrt(w[2])
                    v_tr = v.transpose()
                    self.a11[i, j, k] = v_tr[0, 0]
                    self.a12[i, j, k] = v_tr[0, 1]
                    self.a13[i, j, k] = v_tr[0, 2]
                    self.a21[i, j, k] = v_tr[1, 0]
                    self.a22[i, j, k] = v_tr[1, 1]
                    self.a23[i, j, k] = v_tr[1, 2]
                    self.a31[i, j, k] = v_tr[2, 0]
                    self.a32[i, j, k] = v_tr[2, 1]
                    self.a33[i, j, k] = v_tr[2, 2]

    def _compute_random_values(self):
        self.k1 = np.random.normal(0, 0.5, self.mode_num)
        self.k2 = np.random.normal(0, 0.5, self.mode_num)
        self.k3 = np.random.normal(0, 0.5, self.mode_num)
        self.zeta1 = np.random.normal(0, 1, self.mode_num)
        self.zeta2 = np.random.normal(0, 1, self.mode_num)
        self.zeta3 = np.random.normal(0, 1, self.mode_num)
        self.xi1 = np.random.normal(0, 1, self.mode_num)
        self.xi2 = np.random.normal(0, 1, self.mode_num)
        self.xi3 = np.random.normal(0, 1, self.mode_num)
        self.omega = np.random.normal(0, 1, self.mode_num)

        self.p1 = self.zeta2 * self.k3 - self.zeta3 * self.k2
        self.p2 = self.zeta3 * self.k1 - self.zeta1 * self.k3
        self.p3 = self.zeta1 * self.k2 - self.zeta2 * self.k1

        self.q1 = self.xi2 * self.k3 - self.xi3 * self.k2
        self.q2 = self.xi3 * self.k1 - self.xi1 * self.k3
        self.q3 = self.xi1 * self.k2 - self.xi2 * self.k1

    def _compute_aux_data(self):
        self._compute_eigen_values()
        self._compute_random_values()

    def get_pulsation_at_node(self, i: int, j: int, k: int, time: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v1 = np.zeros(time.shape)
        v2 = np.zeros(time.shape)
        v3 = np.zeros(time.shape)
        for n, t in enumerate(time):
            k1_p = self.k1 * (self.l_t / self.tau_t) / self.c1[i, j, k]
            k2_p = self.k2 * (self.l_t / self.tau_t) / self.c2[i, j, k]
            k3_p = self.k3 * (self.l_t / self.tau_t) / self.c3[i, j, k]
            modes1 = self.p1 * np.cos(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            ) + self.q1 * np.sin(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            )
            modes2 = self.p2 * np.cos(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            ) + self.q2 * np.sin(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            )
            modes3 = self.p3 * np.cos(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            ) + self.q3 * np.sin(
                (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
                 k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * t / self.tau_t
            )
            v1[n] = np.sqrt(2 / self.mode_num) * modes1.sum()
            v2[n] = np.sqrt(2 / self.mode_num) * modes2.sum()
            v3[n] = np.sqrt(2 / self.mode_num) * modes3.sum()
        w1 = self.c1[i, j, k] * v1
        w2 = self.c2[i, j, k] * v2
        w3 = self.c3[i, j, k] * v3
        u1 = self.a11[i, j, k] * w1 + self.a12[i, j, k] * w2 + self.a13[i, j, k] * w3
        u2 = self.a21[i, j, k] * w1 + self.a22[i, j, k] * w2 + self.a23[i, j, k] * w3
        u3 = self.a31[i, j, k] * w1 + self.a32[i, j, k] * w2 + self.a33[i, j, k] * w3
        return u1, u2, u3

    def get_velocity_field(self, time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vel = smirnov_compute_velocity_field(
            self.k1, self.k2, self.k3, self.p1, self.p2, self.p3,
            self.q1, self.q2, self.q3, self.omega, self.c1, self.c2, self.c3,
            self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32, self.a33,
            self.block.mesh[0], self.block.mesh[1], self.block.mesh[2], self.l_t, self.tau_t, self.mode_num,
            time
        )
        u1 = vel[0]
        u2 = vel[1]
        u3 = vel[2]
        # v1 = np.zeros(self.block.shape)
        # v2 = np.zeros(self.block.shape)
        # v3 = np.zeros(self.block.shape)
        # for i in range(self.block.shape[0]):
        #     for j in range(self.block.shape[1]):
        #         for k in range(self.block.shape[2]):
        #             k1_p = self.k1 * (self.l_t / self.tau_t) / self.c1[i, j, k]
        #             k2_p = self.k2 * (self.l_t / self.tau_t) / self.c2[i, j, k]
        #             k3_p = self.k3 * (self.l_t / self.tau_t) / self.c3[i, j, k]
        #             modes1 = self.p1 * np.cos(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             ) + self.q1 * np.sin(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             )
        #             modes2 = self.p2 * np.cos(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             ) + self.q2 * np.sin(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             )
        #             modes3 = self.p3 * np.cos(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             ) + self.q3 * np.sin(
        #                 (k1_p * self.block.mesh[0][i, j, k] + k2_p * self.block.mesh[1][i, j, k] +
        #                  k3_p * self.block.mesh[2][i, j, k]) / self.l_t + self.omega * time / self.tau_t
        #             )
        #             v1[i, j, k] = np.sqrt(2 / self.mode_num) * modes1.sum()
        #             v2[i, j, k] = np.sqrt(2 / self.mode_num) * modes2.sum()
        #             v3[i, j, k] = np.sqrt(2 / self.mode_num) * modes3.sum()
        # w1 = self.c1 * v1
        # w2 = self.c2 * v2
        # w3 = self.c3 * v3
        # u1 = self.a11 * w1 + self.a12 * w2 + self.a13 * w3
        # u2 = self.a21 * w1 + self.a22 * w2 + self.a23 * w3
        # u3 = self.a31 * w1 + self.a32 * w2 + self.a33 * w3
        return u1, u2, u3

    @classmethod
    def _get_energy_desired(cls, k):
        return 16 * (2 / np.pi) ** 0.5 * k**4 * np.exp(-2 * k**2)




