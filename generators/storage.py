from generators.abstract import BCType, Generator, Block
from typing import List, Tuple
import numpy as np
import numpy.linalg as la
from random import choices
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from generators.tools import smirnov_compute_velocity_field, smirnov_compute_pulsation, \
    davidson_compute_velocity_field, davidson_compute_velocity_pulsation, \
    original_sem_compute_velocity_field, original_sem_compute_pulsation


class Lund(Generator):
    def __init__(
            self, block: Block, u_av: Tuple[float, float, float],
            re_xx: float, re_yy: float, re_zz: float,
            re_xy: float, re_xz: float, re_yz: float,
            time_arr: np.ndarray
    ):
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, time_arr)

    def _compute_cholesky(self):
        """Вычисление разложения тензора рейнольдсовых напряжений по Холецкому в каждой точке."""
        self.a11 = np.sqrt(self.re_xx)
        self.a12 = 0
        self.a13 = 0
        self.a21 = self.re_xy / self.a11
        self.a22 = np.sqrt(self.re_yy - self.a21**2)
        self.a23 = 0
        self.a31 = self.re_xz / self.a11
        self.a32 = (self.re_yz - self.a21 * self.a31) / self.a22
        self.a33 = np.sqrt(self.re_zz - self.a31**2 - self.a32**2)

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
        u = self.a11 * u_prime + self.a12 * v_prime + self.a13 * w_prime
        v = self.a21 * u_prime + self.a22 * v_prime + self.a23 * w_prime
        w = self.a31 * u_prime + self.a32 * v_prime + self.a33 * w_prime
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
    def __init__(self, block: Block, u_av: Tuple[float, float, float], l_t: float, tau_t: float,
                 re_xx: float, re_yy: float, re_zz: float,
                 re_xy: float, re_xz: float, re_yz: float,
                 time_arr: np.ndarray,
                 mode_num: int=100,):
        self.tau_t = tau_t
        self.l_t = l_t
        self.mode_num = mode_num
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, time_arr)

    def _compute_eigen_values(self):
        mat = np.array([
            [self.re_xx, self.re_xy, self.re_xz],
            [self.re_xy, self.re_yy, self.re_yz],
            [self.re_xz, self.re_yz, self.re_zz],
        ])
        w, v = la.eig(mat)
        self.c1 = np.sqrt(w[0])
        self.c2 = np.sqrt(w[1])
        self.c3 = np.sqrt(w[2])
        self.a11 = v[0, 0]
        self.a12 = v[0, 1]
        self.a13 = v[0, 2]
        self.a21 = v[1, 0]
        self.a22 = v[1, 1]
        self.a23 = v[1, 2]
        self.a31 = v[2, 0]
        self.a32 = v[2, 1]
        self.a33 = v[2, 2]

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

    def _compute_aux_data_common(self):
        self._compute_random_values()

    def _compute_aux_data_space(self):
        self._compute_eigen_values()

    def compute_aux_data_time_field(self):
        pass

    def compute_aux_data_time_puls(self):
        pass

    def compute_pulsation_at_node(self):
        i, j, k = self.get_puls_node()
        vel = smirnov_compute_pulsation(
            self.k1, self.k2, self.k3, self.p1, self.p2, self.p3,
            self.q1, self.q2, self.q3, self.omega,
            self.c1, self.c2, self.c3,
            self.a11, self.a12, self.a13,
            self.a21, self.a22, self.a23,
            self.a31, self.a32, self.a33,
            self.block.mesh[0][i, j, k], self.block.mesh[1][i, j, k], self.block.mesh[2][i, j, k],
            self.l_t, self.tau_t, self.mode_num, self.time_arr_puls
        )
        self._vel_puls = (vel[0], vel[1], vel[2])

    def compute_velocity_field(self):
        for time in self.time_arr_field:
            vel = smirnov_compute_velocity_field(
                self.k1, self.k2, self.k3, self.p1, self.p2, self.p3,
                self.q1, self.q2, self.q3, self.omega, self.c1, self.c2, self.c3,
                self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32, self.a33,
                self.block.mesh[0], self.block.mesh[1], self.block.mesh[2], self.l_t, self.tau_t, self.mode_num,
                time
            )
            self._vel_field.append((vel[0], vel[1], vel[2]))

    def _get_energy_desired(self, k):
        return 16 * (2 / np.pi) ** 0.5 * k**4 * np.exp(-2 * k**2)


class Davidson(Generator):
    def __init__(
            self, block: Block, u_av: Tuple[float, float, float],
            tau_t: float, l_t: float, dissip_rate: float, viscosity: float, k_t: float, num_modes: int,
            re_xx: float, re_yy: float, re_zz: float,
            re_xy: float, re_xz: float, re_yz: float,
            time_arr: np.ndarray,
    ):
        self.tau_t = tau_t
        self.l_t = l_t
        self.k_t = k_t
        assert k_t == (re_xx + re_yy + re_zz) / 2, 'The value of the kinetic energy of turbulence is not ' \
                                                   'consistent with the values of Reynolds stresses.'
        self.dissip_rate = dissip_rate
        self.viscosity = viscosity
        self.num_modes = num_modes
        self._time_step_field: float = None
        self._time_step_puls: float = None
        self.a_field: float = None
        self.a_puls: float = None
        self.b_field: float = None
        self.b_puls: float = None
        self.phi_field: List[np.ndarray] = []
        self.psi_field: List[np.ndarray] = []
        self.alpha_field: List[np.ndarray] = []
        self.theta_field: List[np.ndarray] = []
        self.k1_field: List[np.ndarray] = []
        self.k2_field: List[np.ndarray] = []
        self.k3_field: List[np.ndarray] = []
        self.sigma1_field: List[np.ndarray] = []
        self.sigma2_field: List[np.ndarray] = []
        self.sigma3_field: List[np.ndarray] = []
        self.phi_puls: List[np.ndarray] = []
        self.psi_puls: List[np.ndarray] = []
        self.alpha_puls: List[np.ndarray] = []
        self.theta_puls: List[np.ndarray] = []
        self.k1_puls: List[np.ndarray] = []
        self.k2_puls: List[np.ndarray] = []
        self.k3_puls: List[np.ndarray] = []
        self.sigma1_puls: List[np.ndarray] = []
        self.sigma2_puls: List[np.ndarray] = []
        self.sigma3_puls: List[np.ndarray] = []
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, time_arr)

    def _compute_eigen_values(self):
        mat = np.array([
            [self.re_xx, self.re_xy, self.re_xz],
            [self.re_xy, self.re_yy, self.re_yz],
            [self.re_xz, self.re_yz, self.re_zz],
        ])
        w, v = la.eig(mat)
        # нормируем диагонализируемую матрицу напряжений Рейнольдса
        self.c1 = np.sqrt(w[0] * 3 / mat.trace())
        self.c2 = np.sqrt(w[1] * 3 / mat.trace())
        self.c3 = np.sqrt(w[2] * 3 / mat.trace())
        self.a11 = v[0, 0]
        self.a12 = v[0, 1]
        self.a13 = v[0, 2]
        self.a21 = v[1, 0]
        self.a22 = v[1, 1]
        self.a23 = v[1, 2]
        self.a31 = v[2, 0]
        self.a32 = v[2, 1]
        self.a33 = v[2, 2]

    def _get_k_from_spectrum(self, alpha, delta_min):
        k_e = alpha * 9 * np.pi / (55 * self.l_t)
        k_min = k_e / 2
        k_max = 2 * np.pi / (2 * delta_min)
        delta_k = (k_max - k_min) / (self.num_modes - 1)
        k_eta = self.dissip_rate ** 0.25 / self.viscosity ** 0.75
        k_arr = np.linspace(k_min, k_max, self.num_modes)
        u_rms = (2 / 3 * self.k_t) ** 0.5
        energy = (
                alpha * u_rms ** 2 / k_e * (k_arr / k_e) ** 4 *
                np.exp(-2 * (k_arr / k_eta) ** 2) / (1 + (k_arr / k_e) ** 2) ** (17 / 6)
        )
        return (energy * delta_k).sum()

    def _compute_wave_numbers_and_amplitude(self):
        # определение минимального шага из предположения об ортогональности и неравномерности блока
        x = self.block.mesh[0][:, 0, 0]
        y = self.block.mesh[1][0, :, 0]
        z = self.block.mesh[2][0, 0, :]

        n_i = self.block.shape[0]
        n_j = self.block.shape[1]
        n_k = self.block.shape[2]

        delta_x = x[1: n_i] - x[0: n_i - 1]
        delta_y = y[1: n_j] - y[0: n_j - 1]
        delta_z = z[1: n_k] - z[0: n_k - 1]

        self.delta_min = min(
            delta_x.min(), delta_y.min(), delta_z.min()
        )

        self.alpha = fsolve(
            lambda x: [self._get_k_from_spectrum(x[0], self.delta_min) - self.k_t], x0=np.array([1.4])
        )[0]
        self.alpha = 1.483
        self.k_e = self.alpha * 9 * np.pi / (55 * self.l_t)
        self.k_min = self.k_e / 2

        self.k_max = 2 * np.pi / (2 * self.delta_min)
        self.delta_k = (self.k_max - self.k_min) / (self.num_modes - 1)
        self.k_arr = np.linspace(self.k_min, self.k_max, self.num_modes)

        self.k_eta = self.dissip_rate**0.25 / self.viscosity**0.75
        self.u_rms = (2 / 3 * self.k_t) ** 0.5
        self.energy = (
                self.alpha * self.u_rms ** 2 / self.k_e * (self.k_arr / self.k_e) ** 4 *
                np.exp(-2 * (self.k_arr / self.k_eta) ** 2) / (1 + (self.k_arr / self.k_e) ** 2) ** (17 / 6)
        )
        self.u_abs = np.sqrt(self.energy * self.delta_k)

    def compute_aux_data_time_field(self):
        if len(self.time_arr_field) > 1:
            self._time_step_field = self.time_arr_field.max() / (self.time_arr_field.shape[0] - 1)
            self.a_field = np.exp(-self._time_step_field / self.tau_t)
        else:
            self.a_field = 1
        self.b_field = (1 - self.a_field**2)**0.5

        for n, time in enumerate(self.time_arr_field):
            uniform_pop = np.linspace(0, 2 * np.pi, 100)
            uniform_weights = [1 / (2 * np.pi) for _ in uniform_pop]
            self.phi_field.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            self.psi_field.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            self.alpha_field.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            theta_pop = np.linspace(0, np.pi, 1000)
            theta_wheights = 1 / 2 * np.sin(theta_pop)
            self.theta_field.append(np.array(choices(theta_pop, theta_wheights, k=self.num_modes)))
            self.k1_field.append(np.sin(self.theta_field[n]) * np.cos(self.phi_field[n]) * self.k_arr)
            self.k2_field.append(np.sin(self.theta_field[n]) * np.sin(self.phi_field[n]) * self.k_arr)
            self.k3_field.append(np.cos(self.theta_field[n]) * self.k_arr)
            self.sigma1_field.append(
                    np.cos(self.phi_field[n]) * np.cos(self.theta_field[n]) * np.cos(self.alpha_field[n]) -
                    np.sin(self.phi_field[n]) * np.sin(self.alpha_field[n])
            )
            self.sigma2_field.append(
                    np.sin(self.phi_field[n]) * np.cos(self.theta_field[n]) * np.cos(self.alpha_field[n]) +
                    np.cos(self.phi_field[n]) * np.sin(self.alpha_field[n])
            )
            self.sigma3_field.append(-np.sin(self.theta_field[n]) * np.cos(self.alpha_field[n]))

    def compute_aux_data_time_puls(self):
        if len(self.time_arr_puls) > 1:
            self._time_step_puls = self.time_arr_puls.max() / (self.time_arr_puls.shape[0] - 1)
            self.a_puls = np.exp(-self._time_step_puls / self.tau_t)
        else:
            self.a_puls = 1
        self.b_puls = (1 - self.a_puls ** 2) ** 0.5

        for n, time in enumerate(self.time_arr_puls):
            uniform_pop = np.linspace(0, 2 * np.pi, 100)
            uniform_weights = [1 / (2 * np.pi) for _ in uniform_pop]
            self.phi_puls.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            self.psi_puls.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            self.alpha_puls.append(np.array(choices(uniform_pop, uniform_weights, k=self.num_modes)))
            theta_pop = np.linspace(0, np.pi, 1000)
            theta_wheights = 1 / 2 * np.sin(theta_pop)
            self.theta_puls.append(np.array(choices(theta_pop, theta_wheights, k=self.num_modes)))
            self.k1_puls.append(np.sin(self.theta_puls[n]) * np.cos(self.phi_puls[n]))
            self.k2_puls.append(np.sin(self.theta_puls[n]) * np.sin(self.phi_puls[n]))
            self.k3_puls.append(np.cos(self.theta_puls[n]))
            self.sigma1_puls.append(
                    np.cos(self.phi_puls[n]) * np.cos(self.theta_puls[n]) * np.cos(self.alpha_puls[n]) -
                    np.sin(self.phi_puls[n]) * np.sin(self.alpha_puls[n])
            )
            self.sigma2_puls.append(
                    np.sin(self.phi_puls[n]) * np.cos(self.theta_puls[n]) * np.cos(self.alpha_puls[n]) +
                    np.cos(self.phi_puls[n]) * np.sin(self.alpha_puls[n])
            )
            self.sigma3_puls.append(-np.sin(self.theta_puls[n]) * np.cos(self.alpha_puls[n]))

    def _compute_aux_data_common(self):
        self._compute_wave_numbers_and_amplitude()

    def _compute_aux_data_space(self):
        self._compute_eigen_values()

    def compute_velocity_field(self):
        shape = self.block.shape
        for n, time in enumerate(self.time_arr_field):
            if n == 0:
                vel = davidson_compute_velocity_field(
                    self.k1_field[n], self.k2_field[n], self.k3_field[n],
                    self.sigma1_field[n], self.sigma2_field[n], self.sigma3_field[n], self.psi_field[n],
                    self.c1, self.c2, self.c3,
                    self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32, self.a33,
                    self.block.mesh[0], self.block.mesh[1], self.block.mesh[2],
                    u_abs=self.u_abs, a=self.a_field, b=1,
                    v1_previous=np.zeros(shape), v2_previous=np.zeros(shape), v3_previous=np.zeros(shape)
                )
                self._vel_field.append((vel[0], vel[1], vel[2]))
            else:
                vel = davidson_compute_velocity_field(
                    self.k1_field[n], self.k2_field[n], self.k3_field[n],
                    self.sigma1_field[n], self.sigma2_field[n], self.sigma3_field[n], self.psi_field[n],
                    self.c1, self.c2, self.c3,
                    self.a11, self.a12, self.a13, self.a21, self.a22, self.a23, self.a31, self.a32, self.a33,
                    self.block.mesh[0], self.block.mesh[1], self.block.mesh[2],
                    u_abs=self.u_abs, a=self.a_field, b=self.b_field,
                    v1_previous=self._vel_field[n - 1][0],
                    v2_previous=self._vel_field[n - 1][1],
                    v3_previous=self._vel_field[n - 1][2]
                )
                self._vel_field.append((vel[0], vel[1], vel[2]))

    def compute_pulsation_at_node(self):
        i, j, k = self.get_puls_node()
        vel = davidson_compute_velocity_pulsation(
            self.k1_puls, self.k2_puls, self.k3_puls, self.sigma1_puls, self.sigma2_puls, self.sigma3_puls,
            self.psi_puls, self.c1, self.c2, self.c3,
            self.a11, self.a12, self.a13,
            self.a21, self.a22, self.a23,
            self.a31, self.a32, self.a33,
            self.block.mesh[0][i, j, k], self.block.mesh[1][i, j, k], self.block.mesh[2][i, j, k],
            self.u_abs, self.a_puls, self.b_puls,
            self.time_arr_puls
        )
        self._vel_puls = (vel[0], vel[1], vel[2])

    def _get_energy_desired(self, k):
        return float(interp1d(self.k_arr, self.energy, fill_value=0, bounds_error=False)(k))


class OriginalSEM(Generator):
    def __init__(
            self, block: Block, u_av: Tuple[float, float, float], sigma: float, eddy_num: int,
            re_xx: float, re_yy: float, re_zz: float,
            re_xy: float, re_xz: float, re_yz: float,
            time_arr: np.ndarray,
    ):
        self.sigma = sigma
        self.eddy_num = eddy_num
        self.eddy_positions_field: np.ndarray = []
        self.eddy_positions_puls: np.ndarray = []
        Generator.__init__(self, block, u_av, re_xx, re_yy, re_zz, re_xy, re_xz, re_yz, time_arr)

    def _compute_cholesky(self):
        """Вычисление разложения тензора рейнольдсовых напряжений по Холецкому в каждой точке."""
        self.a11 = np.sqrt(self.re_xx)
        self.a12 = 0
        self.a13 = 0
        self.a21 = self.re_xy / self.a11
        self.a22 = np.sqrt(self.re_yy - self.a21 ** 2)
        self.a23 = 0
        self.a31 = self.re_xz / self.a11
        self.a32 = (self.re_yz - self.a21 * self.a31) / self.a22
        self.a33 = np.sqrt(self.re_zz - self.a31 ** 2 - self.a32 ** 2)

    def _compute_init_eddies_pos(self):
        self.u0 = self.u_av[0]
        self.v0 = self.u_av[1]
        self.w0 = self.u_av[2]
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
        :param time: Временной интервал, на котором нужно вычислить позиции вихрей
        :param num_ts: Число отрезков на этом интервале.

        Возвращает набор значений координат позиций вихрей и их интенсивностей в различные моменты времени.
        """
        if num_ts != 0:
            dt = time / num_ts
        else:
            dt = 0
        in_planes = self.get_in_planes(
            self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
            self.x_e_init, self.y_e_init, self.z_e_init, self.u0, self.v0, self.w0,
        )
        x_e = self.x_e_init
        y_e = self.y_e_init
        z_e = self.z_e_init
        epsilon = self.epsilon_init
        res = np.zeros((num_ts + 1, 6, self.eddy_num))
        for n in range(num_ts + 1):
            if n > 0:
                x_e = x_e + dt * self.u0
                y_e = y_e + dt * self.v0
                z_e = z_e + dt * self.w0
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

            res[n, 0, :] = x_e
            res[n, 1, :] = y_e
            res[n, 2, :] = z_e
            res[n, 3, :] = epsilon[:, 0]
            res[n, 4, :] = epsilon[:, 1]
            res[n, 5, :] = epsilon[:, 2]
        return res

    def _compute_aux_data_common(self):
        self._compute_init_eddies_pos()

    def _compute_aux_data_space(self):
        self._compute_cholesky()

    def compute_aux_data_time_field(self):
        self.eddy_positions_field = self.get_eddies_params(
            self.time_arr_field.max() - self.time_arr_field.min(),
            self.time_arr_field.shape[0] - 1
        )

    def compute_aux_data_time_puls(self):
        self.eddy_positions_puls = self.get_eddies_params(
            self.time_arr_puls.max() - self.time_arr_puls.min(),
            self.time_arr_puls.shape[0] - 1
        )

    @classmethod
    def form_func(cls, x):
        return (np.abs(x) < 1) * (np.sqrt(1.5) * (1 - np.abs(x)))

    def compute_velocity_field(self):
        vel = original_sem_compute_velocity_field(
            positions=self.eddy_positions_field,
            x=self.block.mesh[0], y=self.block.mesh[1], z=self.block.mesh[2],
            sigma=self.sigma, volume=self.volume, eddy_num=self.eddy_num,
            a11=self.a11, a12=self.a12, a13=self.a13,
            a21=self.a21, a22=self.a22, a23=self.a23,
            a31=self.a31, a32=self.a32, a33=self.a33,
        )
        for i in range(vel.shape[0]):
            self._vel_field.append((vel[i, 0, :, :, :], vel[i, 1, :, :, :], vel[i, 2, :, :, :]))

    def compute_pulsation_at_node(self):
        i, j, k = self.get_puls_node()
        self._vel_puls = original_sem_compute_pulsation(
            positions=self.eddy_positions_puls,
            x=self.block.mesh[0][i, j, k], y=self.block.mesh[1][i, j, k], z=self.block.mesh[2][i, j, k],
            sigma=self.sigma, volume=self.volume, eddy_num=self.eddy_num,
            a11=self.a11, a12=self.a12, a13=self.a13,
            a21=self.a21, a22=self.a22, a23=self.a23,
            a31=self.a31, a32=self.a32, a33=self.a33,
        )





