from generators.abstract import Generator
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


class Analyzer:
    def __init__(self, generator: Generator):
        self.generator = generator

    def plot_2d_velocity_field(self, figsize=(7, 7), num_levels=20, vmin=-3.5, vmax=3.5, grid=True):
        x = self.generator.block.mesh[0][:, :, 0]
        y = self.generator.block.mesh[1][:, :, 0]
        u3d, v3d, w3d = self.generator.get_velocity_field(0)
        u = u3d[:, :, 0]
        v = v3d[:, :, 0]
        w = w3d[:, :, 0]

        plt.figure(figsize=figsize)
        plt.contourf(x, y, u, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        plt.title('U', fontsize=16)
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        plt.show()

        plt.figure(figsize=figsize)
        plt.contourf(x, y, v, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        plt.title('V', fontsize=16)
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        plt.show()

        plt.figure(figsize=figsize)
        plt.contourf(x, y, w, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        plt.title('W', fontsize=16)
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        plt.show()

    def save_velocity_field_tec(self, fname):
        pass

    def plot_velocity_history(self, i: int, j: int, k: int, ts, num_ts: int, figsize=(7, 7)):
        t_arr = np.arange(0, ts*num_ts, ts)
        u, v, w = self.generator.get_pulsation_at_node(i, j, k, t_arr)
        plt.figure(figsize=figsize)
        plt.plot(t_arr, u, color='red', lw=1)
        plt.grid()
        plt.xlim(xmin=0, xmax=ts*num_ts)
        plt.title('U', fontsize=16)
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t_arr, v, color='red', lw=1)
        plt.grid()
        plt.xlim(xmin=0, xmax=ts * num_ts)
        plt.title('V', fontsize=16)
        plt.show()

        plt.figure(figsize=figsize)
        plt.plot(t_arr, w, color='red', lw=1)
        plt.grid()
        plt.xlim(xmin=0, xmax=ts * num_ts)
        plt.title('W', fontsize=16)
        plt.show()

    @classmethod
    def _get_average_arr(cls, value: np.ndarray):
        res = []
        sum = 0
        for n, i in enumerate(value):
            sum += i
            res.append(sum / (n + 1))
        return np.array(res)

    @classmethod
    def _get_average(cls, value: np.ndarray):
        return value.sum() / len(value)

    def plot_moments(self, i: int, j: int, k: int, ts: float, num_ts: int, figsize=(7, 7)):
        t_arr = np.arange(0, ts * num_ts, ts)
        u, v, w = self.generator.get_pulsation_at_node(i, j, k, t_arr)
        uu_av = self._get_average_arr(u * u)
        vv_av = self._get_average_arr(v * v)
        ww_av = self._get_average_arr(w * w)
        uv_av = self._get_average_arr(u * v)
        uw_av = self._get_average_arr(u * w)
        vw_av = self._get_average_arr(v * w)
        plt.figure(figsize=figsize)
        plt.plot(t_arr, uu_av, lw=1, color='red', label=r'$<v_x^2>$')
        plt.plot(t_arr, vv_av, lw=1, color='blue', label=r'$<v_y^2>$')
        plt.plot(t_arr, ww_av, lw=1, color='green', label=r'$<v_z^2>$')
        plt.plot(t_arr, uv_av, lw=1, color='red', ls=':', label=r'$<v_x v_y>$')
        plt.plot(t_arr, uw_av, lw=1, color='blue', ls=':', label=r'$<v_x v_z>$')
        plt.plot(t_arr, vw_av, lw=1, color='green', ls=':', label=r'$<v_y v_z>$')
        plt.legend(fontsize=12)
        plt.xlim(0, num_ts*ts)
        # plt.ylim(-0.5, 1.5)
        plt.grid()
        plt.show()

    def plot_divergence_field_2d(self, figzie=(7, 7), num_levels=20, vmin=-300, vmax=300, grid=True):
        x = self.generator.block.mesh[0][:, :, 0]
        y = self.generator.block.mesh[1][:, :, 0]
        vel = self.generator.get_velocity_field(0)
        div = self.generator.get_divergence(vel, self.generator.block.mesh, self.generator.block.shape)
        div_2d = div[:, :, 0]

        plt.figure(figsize=figzie)
        plt.contourf(x, y, div_2d, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        plt.title(r'$div(U)$', fontsize=16)
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        plt.show()

    def save_divergence_field_tec(self, fname):
        pass

    def plot_two_point_space_correlation(self, i0: int, j0: int, k0: int,
                                         ts: float, num_ts: int,
                                         di: int=1, dj: int=1, dk: int=1, num: int=20,
                                         figsize=(7, 7)):
        t_arr = np.arange(0, ts * num_ts, ts)
        r = np.zeros(num)
        cor_uu = np.zeros(num)
        cor_vv = np.zeros(num)
        cor_ww = np.zeros(num)
        u0, v0, w0 = self.generator.get_pulsation_at_node(i0, j0, k0, t_arr)
        u0u0_av = self._get_average(u0 * u0)
        v0v0_av = self._get_average(v0 * v0)
        w0w0_av = self._get_average(w0 * w0)

        for n in range(1, num):
            i = i0 + di * n
            j = j0 + dj * n
            k = k0 + dk * n
            u, v, w = self.generator.get_pulsation_at_node(i, j, k, t_arr)
            r[n] = np.sqrt((self.generator.block.mesh[0][i, j, k] - self.generator.block.mesh[0][i0, j0, k0])**2 +
                           (self.generator.block.mesh[1][i, j, k] - self.generator.block.mesh[1][i0, j0, k0])**2 +
                           (self.generator.block.mesh[2][i, j, k] - self.generator.block.mesh[2][i0, j0, k0])**2)
            uu_av = self._get_average(u * u)
            vv_av = self._get_average(v * v)
            ww_av = self._get_average(w * w)
            u0u_av = self._get_average(u0 * u)
            v0v_av = self._get_average(v0 * v)
            w0w_av = self._get_average(w0 * w)
            cor_uu[n] = u0u_av / (np.sqrt(u0u0_av) * np.sqrt(uu_av))
            cor_vv[n] = v0v_av / (np.sqrt(v0v0_av) * np.sqrt(vv_av))
            cor_ww[n] = w0w_av / (np.sqrt(w0w0_av) * np.sqrt(ww_av))

        plt.figure(figsize=figsize)
        plt.plot(r, cor_uu, color='red', lw=1.5, label=r'$R_{xx}^r$')
        plt.plot(r, cor_vv, color='blue', lw=1.5, label=r'$R_{yy}^r$')
        plt.plot(r, cor_ww, color='green', lw=1.5, label=r'$R_{zz}^r$')
        plt.grid()
        plt.xlim(xmin=0, xmax=ts*num_ts)
        plt.ylim(ymin=-1.1, ymax=1.1)
        plt.xlabel(r'$r,\ м$', fontsize=14)
        plt.legend(fontsize=12)
        plt.show()

    def plot_two_point_time_correlation(self, i: int, j: int, k: int,
                                        t0: float, t1: float, t2: float, num_dt: int=50, num_av: int=100,
                                        figsize=(7, 7)):
        # Осреднение будет происходить от момента t0 до момента t1 + dt_arr[i] для определения
        # i-ого значения коэффициента автокорреляции
        dt_arr = np.linspace(0, t2 - t1, num_dt)
        cor_uu = np.zeros(num_dt)
        cor_vv = np.zeros(num_dt)
        cor_ww = np.zeros(num_dt)
        t0_arr = np.linspace(t0, t1, num_av)
        u0, v0, w0 = self.generator.get_pulsation_at_node(i, j, k, t0_arr)
        u0u0_av = self._get_average(u0 * u0)
        v0v0_av = self._get_average(v0 * v0)
        w0w0_av = self._get_average(w0 * w0)
        for n in range(1, num_dt):
            t_arr = np.linspace(t0, t1 + dt_arr[n], num_av)
            u, v, w = self.generator.get_pulsation_at_node(i, j, k, t_arr)
            uu_av = self._get_average(u * u)
            vv_av = self._get_average(v * v)
            ww_av = self._get_average(w * w)
            u0u_av = self._get_average(u0 * u)
            v0v_av = self._get_average(v0 * v)
            w0w_av = self._get_average(w0 * w)
            cor_uu[n] = u0u_av / (np.sqrt(u0u0_av) * np.sqrt(uu_av))
            cor_vv[n] = v0v_av / (np.sqrt(v0v0_av) * np.sqrt(vv_av))
            cor_ww[n] = w0w_av / (np.sqrt(w0w0_av) * np.sqrt(ww_av))

        plt.figure(figsize=figsize)
        plt.plot(dt_arr, cor_uu, color='red', lw=1.5, label=r'$R_{xx}^t$')
        plt.plot(dt_arr, cor_vv, color='blue', lw=1.5, label=r'$R_{yy}^t$')
        plt.plot(dt_arr, cor_ww, color='green', lw=1.5, label=r'$R_{zz}^t$')
        plt.grid()
        plt.xlim(xmin=0, xmax=dt_arr.max())
        plt.ylim(ymin=-1.1, ymax=1.1)
        plt.xlabel(r'$\Delta t,\ с$', fontsize=14)
        plt.legend(fontsize=12)
        plt.show()

    @classmethod
    def _get_energy(cls, m_grid: np.ndarray, energy_arr: np.ndarray, m_mag):
        """Вычисляет величину энергии в шаровом слое единичной толщины в простанстве m"""
        energy_arr_filt = energy_arr[(m_grid > m_mag - 0.5) * (m_grid < m_mag + 0.5)]
        energy = energy_arr_filt.sum()
        return energy

    @classmethod
    def get_spectrum_2d(cls, x: np.ndarray, vel: np.ndarray, num_pnt=100):
        """Вычисление двухмерного энергетического спектра для компоненты скорости на квадратной сетке."""
        vel_f = np.fft.fftn(vel)
        vel_f = np.fft.fftshift(vel_f)
        length = x.max() - x.min()
        m1, m2 = np.meshgrid(
            np.linspace(-x.shape[0] / 2, x.shape[0] / 2 - 1, x.shape[0]),
            np.linspace(-x.shape[0] / 2, x.shape[0] / 2 - 1, x.shape[0]),
        )
        m_grid = np.sqrt(m1**2 + m2**2)
        energy = 0.5 * (np.abs(vel_f) / (x.shape[0]**2)) ** 2

        m_mag = np.linspace(0, m_grid.max(), num_pnt)
        e_k_mag = np.zeros(num_pnt)
        for i in range(num_pnt):
            e_k_mag[i] = cls._get_energy(m_grid, energy, m_mag[i]) * length / (2 * np.pi)
        k_mag = m_mag * 2 * np.pi / length
        return k_mag, e_k_mag

    def plot_spectrum_2d(self, figsize=(7, 7), num_pnt=100):
        if self.generator.block.shape[0] != self.generator.block.shape[1]:
            raise Exception('Для построения спектра блок должен быть квадратным')
        u3d, v3d, w3d = self.generator.get_velocity_field(0)
        u = u3d[:, :, 0]
        v = v3d[:, :, 0]
        w = w3d[:, :, 0]
        x = self.generator.block.mesh[0][:, 0, 0]
        k_u, e_u = self.get_spectrum_2d(x, u, num_pnt)
        k_v, e_v = self.get_spectrum_2d(x, v, num_pnt)
        k_w, e_w = self.get_spectrum_2d(x, w, num_pnt)

        plt.figure(figsize=figsize)
        plt.plot(k_u, e_u, color='red', label='u', lw=1.5)
        plt.plot(k_v, e_v, color='blue', label='v', lw=1.5)
        plt.plot(k_w, e_w, color='green', label='w', lw=1.5)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(which='both')
        plt.legend(fontsize=12)
        plt.show()















