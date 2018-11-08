from generators.abstract import BCType, Generator, Block
from typing import List, Tuple
import numpy as np
import numpy.linalg as la
from generators.tools import smirnov_compute_velocity_field


class Lund(Generator):
    def __init__(
            self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float,
            re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
            re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray,
            time_arr: np.ndarray
    ):
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, time_arr)
        self.l_t = l_t

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

    def get_pulsation_at_node(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._vel_puls

    def get_velocity_field(self, time_step, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._vel_field[time_step]

    def compute_velocity_field(self):
        for time in self.time_arr_field:
            u_prime = np.random.normal(0, 1, self.block.shape)
            v_prime = np.random.normal(0, 1, self.block.shape)
            w_prime = np.random.normal(0, 1, self.block.shape)
            u = self.a11 * u_prime + self.a12 * v_prime + self.a13 * w_prime
            v = self.a21 * u_prime + self.a22 * v_prime + self.a23 * w_prime
            w = self.a31 * u_prime + self.a32 * v_prime + self.a33 * w_prime
            self._vel_field.append((u, v, w))

    def compute_pulsation_at_node(self):
        i = self._i_puls
        j = self._j_puls
        k = self._k_puls
        u_prime = np.random.normal(0, 1, self.time_arr_puls.shape)
        v_prime = np.random.normal(0, 1, self.time_arr_puls.shape)
        w_prime = np.random.normal(0, 1, self.time_arr_puls.shape)
        u = self.a11[i, j, k] * u_prime + self.a12[i, j, k] * v_prime + self.a13[i, j, k] * w_prime
        v = self.a21[i, j, k] * u_prime + self.a22[i, j, k] * v_prime + self.a23[i, j, k] * w_prime
        w = self.a31[i, j, k] * u_prime + self.a32[i, j, k] * v_prime + self.a33[i, j, k] * w_prime
        self._vel_puls = (u, v, w)

    def _compute_aux_data_space(self):
        self._compute_cholesky()

    def _compute_aux_data_common(self):
        pass

    def compute_aux_data_time_puls(self):
        pass

    def compute_aux_data_time_field(self):
        pass


class Smirnov(Generator):
    def __init__(self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float, tau_t: float,
                 re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
                 re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray,
                 mode_num: int=100,):
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz)
        self.tau_t = tau_t
        self.l_t = l_t
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

    def get_velocity_field(self, time_step, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vel = smirnov_compute_velocity_field(
            self.k1, self.k2, self.k3, self.p1, self.p2, self.p3,
            self.q1, self.q2, self.q3, self.omega, self.c1, self.c2, self.c3,
            self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32, self.a33,
            self.block.mesh[0], self.block.mesh[1], self.block.mesh[2], self.l_t, self.tau_t, self.mode_num,
            time_step
        )
        u1 = vel[0]
        u2 = vel[1]
        u3 = vel[2]
        return u1, u2, u3

    @classmethod
    def _get_energy_desired(cls, k):
        return 16 * (2 / np.pi) ** 0.5 * k**4 * np.exp(-2 * k**2)


class OriginalSEM(Generator):
    def __init__(
            self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], sigma: float, eddy_num: int,
            re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
            re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray
    ):
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz)
        self.sigma = sigma
        self.eddy_num = eddy_num
        self._compute_aux_data()

    def _compute_cholesky(self):
        """Вычисление разложения тензора рейнольдсовых напряжений по Холецкому в каждой точке."""
        self.a11 = np.sqrt(self.re_xx)
        self.a12 = np.zeros(self.block.shape)
        self.a13 = np.zeros(self.block.shape)
        self.a21 = self.re_xy / self.a11
        self.a22 = np.sqrt(self.re_yy - self.a21 ** 2)
        self.a23 = np.zeros(self.block.shape)
        self.a31 = self.re_xz / self.a11
        self.a32 = (self.re_yz - self.a21 * self.a31) / self.a22
        self.a33 = np.sqrt(self.re_zz - self.a31 ** 2 - self.a32 ** 2)

    def _compute_init_eddies_pos(self):
        self.u0 = self.u_av[0].mean()
        self.v0 = self.u_av[1].mean()
        self.w0 = self.u_av[2].mean()
        self.x_min = self.block.mesh[0].min() - self.sigma
        self.x_max = self.block.mesh[0].max() + self.sigma
        self.y_min = self.block.mesh[1].min() - self.sigma
        self.y_max = self.block.mesh[1].max() + self.sigma
        self.z_min = self.block.mesh[2].min() - self.sigma
        self.z_max = self.block.mesh[2].max() + self.sigma
        self.volume = (self.x_max - self.x_min) * (self.y_max - self.y_min) * (self.z_max - self.z_min)
        # self.eddy_num = int(self.volume / self.sigma**3)
        self.x_e_init = np.random.uniform(self.x_min, self.x_max, self.eddy_num)
        self.y_e_init = np.random.uniform(self.y_min, self.y_max, self.eddy_num)
        self.z_e_init = np.random.uniform(self.z_min, self.z_max, self.eddy_num)
        self.epsilon_init = np.random.normal(0, 1, (self.eddy_num, 3))
        # self.epsilon_init = np.random.uniform(-1, 1, (self.eddy_num, 3))

    @classmethod
    def _get_line_plane_intersection(cls, x0, y0, z0, xv, yv, zv, a, b, c, d):
        k = a * xv + b * yv + c * zv
        x_res = (x0 * (b * yv + c * zv) - xv * (b * y0 + c * z0 + d))
        y_res = (y0 * (a * xv + c * zv) - yv * (a * x0 + c * z0 + d))
        z_res = (z0 * (b * yv + a * xv) - zv * (b * y0 + a * x0 + d))
        if k == 0:
            return None
        else:
            return x_res / k, y_res / k, z_res / k

    @classmethod
    def get_scalar_prod(cls, x1, y1, z1, x2, y2, z2):
        return x1 * x2 + y1 * y2 + z1 * z2

    @classmethod
    def get_in_planes(cls, x_min, x_max, y_min, y_max, z_min, z_max,
                      x_e: np.ndarray, y_e: np.ndarray, z_e: np.ndarray, u0, v0, w0):
        # коэффициенты плоскостей, ограничивающих область
        bounds_coef = (
            (1, 0, 0, -x_min), (1, 0, 0, -x_max),
            (0, 1, 0, -y_min), (0, 1, 0, -y_max),
            (0, 0, 1, -z_min), (0, 0, 1, -z_max),
        )
        # интервалы координат, в которых расположена каждая из 6 граничных граней области
        bounds = (
            ((x_min, x_min), (y_min, y_max), (z_min, z_max)),
            ((x_max, x_max), (y_min, y_max), (z_min, z_max)),
            ((x_min, x_max), (y_min, y_min), (z_min, z_max)),
            ((x_min, x_max), (y_max, y_max), (z_min, z_max)),
            ((x_min, x_max), (y_min, y_max), (z_min, z_min)),
            ((x_min, x_max), (y_min, y_max), (z_max, z_max)),
        )
        in_planes = []
        for x0, y0, z0 in zip(x_e, y_e, z_e):
            intersecs = (
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[0]),
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[1]),
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[2]),
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[3]),
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[4]),
                cls._get_line_plane_intersection(x0, y0, z0, u0, v0, w0, *bounds_coef[5]),
            )
            for intersec, bound in zip(intersecs, bounds):
                if intersec:
                    if (
                            (intersec[0] >= x_min or intersec[0] <= x_max) and
                            (intersec[1] >= y_min or intersec[1] <= y_max) and
                            (intersec[2] >= z_min or intersec[2] <= z_max)
                    ):
                        s = cls.get_scalar_prod(
                            (intersec[0] - x0), (intersec[1] - y0), (intersec[2] - z0), u0, v0, w0
                        )
                        if s < 0:
                            in_planes.append(bound)
        return in_planes

    def get_eddies_params(self, time, num_ts: int = 100):
        """
        Возвращает набор значений координат позиций вихрей и их интенсивностей в различные моменты времени.
        """
        dt = time / (num_ts + 1)
        in_planes = self.get_in_planes(
            self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
            self.x_e_init, self.y_e_init, self.z_e_init, self.u0, self.v0, self.w0,
        )
        x_e = self.x_e_init
        y_e = self.y_e_init
        z_e = self.z_e_init
        epsilon = self.epsilon_init
        for _ in range(num_ts):
            for i in range(self.eddy_num):
                if (
                        x_e[i] < self.x_min or x_e[i] > self.x_max or
                        y_e[i] < self.y_min or y_e[i] > self.y_max or
                        z_e[i] < self.z_min or z_e[i] > self.z_max
                ):
                    x_e[i] = np.random.uniform(in_planes[i][0][0], in_planes[i][0][1])
                    y_e[i] = np.random.uniform(in_planes[i][1][0], in_planes[i][1][1])
                    z_e[i] = np.random.uniform(in_planes[i][2][0], in_planes[i][2][1])
                    epsilon[i] = np.random.normal(0, 1, 3)
                    # epsilon[i] = np.random.uniform(-1, 1, 3)

            x_e = x_e + dt * self.u0
            y_e = y_e + dt * self.v0
            z_e = z_e + dt * self.w0
            yield x_e, y_e, z_e, epsilon

    def _compute_aux_data(self):
        self._compute_cholesky()
        self._compute_init_eddies_pos()

    @classmethod
    def form_func(cls, x):
        return (np.abs(x) < 1) * (np.sqrt(1.5) * (1 - np.abs(x)))

    def get_pulsation_at_node(self, i: int, j: int, k: int, time: np.ndarray):
        positions = list(self.get_eddies_params(time.max(), time.shape[0]))
        u = np.zeros(time.shape)
        v = np.zeros(time.shape)
        w = np.zeros(time.shape)
        for n, position in enumerate(positions):
            x_e = np.array(position[0])
            y_e = np.array(position[1])
            z_e = np.array(position[2])
            epsilon = np.array(position[3])
            f_sigma = self.volume**0.5 / self.sigma**1.5 * (
                self.form_func((self.block.mesh[0][i, j, k] - x_e) / self.sigma) *
                self.form_func((self.block.mesh[1][i, j, k] - y_e) / self.sigma) *
                self.form_func((self.block.mesh[2][i, j, k] - z_e) / self.sigma)
            )
            u1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 0]).sum()
            v1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 1]).sum()
            w1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 2]).sum()
            u[n] = self.a11[i, j, k] * u1 + self.a12[i, j, k] * v1 + self.a13[i, j, k] * w1
            v[n] = self.a21[i, j, k] * u1 + self.a22[i, j, k] * v1 + self.a23[i, j, k] * w1
            w[n] = self.a31[i, j, k] * u1 + self.a32[i, j, k] * v1 + self.a33[i, j, k] * w1
        return u, v, w

    def get_velocity_field(self, time_step, **kwargs):
        num_ts = kwargs['num_ts']
        x_e, y_e, z_e, epsilon = list(self.get_eddies_params(time_step, num_ts))[num_ts - 1]
        u = np.zeros(self.block.shape)
        v = np.zeros(self.block.shape)
        w = np.zeros(self.block.shape)
        for i in range(self.block.shape[0]):
            for j in range(self.block.shape[1]):
                for k in range(self.block.shape[2]):
                    f_sigma = np.sqrt(self.volume) / self.sigma ** 1.5 * (
                            self.form_func((self.block.mesh[0][i, j, k] - x_e) / self.sigma) *
                            self.form_func((self.block.mesh[1][i, j, k] - y_e) / self.sigma) *
                            self.form_func((self.block.mesh[2][i, j, k] - z_e) / self.sigma)
                    )
                    u1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 0]).sum()
                    v1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 1]).sum()
                    w1 = 1 / (np.sqrt(self.eddy_num)) * (f_sigma * epsilon[:, 2]).sum()
                    u[i, j, k] = self.a11[i, j, k] * u1 + self.a12[i, j, k] * v1 + self.a13[i, j, k] * w1
                    v[i, j, k] = self.a21[i, j, k] * u1 + self.a22[i, j, k] * v1 + self.a23[i, j, k] * w1
                    w[i, j, k] = self.a31[i, j, k] * u1 + self.a32[i, j, k] * v1 + self.a33[i, j, k] * w1
        return u, v, w




