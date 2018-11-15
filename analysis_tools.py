from generators.abstract import Generator
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


class Analyzer:
    def __init__(self, generator: Generator):
        self.generator = generator
        self.generator.compute_aux_data_time_field()
        self.generator.compute_velocity_field()

    def plot_2d_velocity_field(
            self, figsize=(7, 7), num_levels=20, vmin=-3.5, vmax=3.5, grid=True, title_fsize=18, title=True,
            axes=(0.05, 0.05, 0.9, 0.9), fname=None
    ):
        x = self.generator.block.mesh[0][:, :, 0]
        y = self.generator.block.mesh[1][:, :, 0]
        u3d, v3d, w3d = self.generator.get_velocity_field(0)
        u = u3d[:, :, 0]
        v = v3d[:, :, 0]
        w = w3d[:, :, 0]

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.contourf(x, y, u, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        if title:
            plt.title('U', fontsize=title_fsize, fontweight='bold')
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        if fname:
            plt.savefig(fname + '_U')
        plt.show()

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.contourf(x, y, v, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        if title:
            plt.title('V', fontsize=title_fsize, fontweight='bold')
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        if fname:
            plt.savefig(fname + '_V')
        plt.show()

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.contourf(x, y, w, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        if title:
            plt.title('W', fontsize=title_fsize, fontweight='bold')
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        if fname:
            plt.savefig(fname + '_W')
        plt.show()

    def save_velocity_field_tec(self, fname):
        pass

    def plot_velocity_history(
            self, i: int, j: int, k: int, ts, num_ts: int, figsize=(7, 7), ylim=(-3, 3),
            label_fsize=16, ticks_fsize=12, title_fsize=18, title=True, axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
        t_arr = np.arange(0, ts*num_ts, ts)
        self.generator.set_puls_node(i, j, k)
        self.generator.time_arr_puls = t_arr
        self.generator.compute_aux_data_time_puls()
        self.generator.compute_pulsation_at_node()
        u, v, w = self.generator.get_pulsation_at_node()

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(t_arr, u, color='red', lw=1)
        plt.grid()

        if ylim:
            plt.ylim(*ylim)
        plt.xlim(xmin=0, xmax=ts*num_ts)
        if title:
            plt.title('U', fontsize=title_fsize, fontweight='bold')
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('t, с', fontsize=label_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        if fname:
            plt.savefig(fname + '_U')
        plt.show()

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(t_arr, v, color='red', lw=1)
        plt.grid()

        if ylim:
            plt.ylim(*ylim)
        plt.xlim(xmin=0, xmax=ts * num_ts)
        if title:
            plt.title('V', fontsize=title_fsize, fontweight='bold')
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('t, с', fontsize=label_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        if fname:
            plt.savefig(fname + '_V')
        plt.show()

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(t_arr, w, color='red', lw=1)
        plt.grid()

        if ylim:
            plt.ylim(*ylim)
        plt.xlim(xmin=0, xmax=ts * num_ts)
        if title:
            plt.title('W', fontsize=title_fsize, fontweight='bold')
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('t, с', fontsize=label_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        if fname:
            plt.savefig(fname + '_W')
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

    def plot_moments(
            self, i: int, j: int, k: int, ts: float, num_ts: int, figsize=(7, 7), ylim=(-0.5, 1),
            legend_fsize=14, ticks_fsize=12, label_fsize=14, axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
        t_arr = np.arange(0, ts * num_ts, ts)
        self.generator.set_puls_node(i, j, k)
        self.generator.time_arr_puls = t_arr
        self.generator.compute_aux_data_time_puls()
        self.generator.compute_pulsation_at_node()
        u, v, w = self.generator.get_pulsation_at_node()
        uu_av = self._get_average_arr(u * u)
        vv_av = self._get_average_arr(v * v)
        ww_av = self._get_average_arr(w * w)
        uv_av = self._get_average_arr(u * v)
        uw_av = self._get_average_arr(u * w)
        vw_av = self._get_average_arr(v * w)

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(t_arr, uu_av, lw=1.5, color='red', label=r'$<v_x^2>$')
        plt.plot(t_arr, vv_av, lw=1.5, color='blue', label=r'$<v_y^2>$')
        plt.plot(t_arr, ww_av, lw=1.5, color='green', label=r'$<v_z^2>$')
        plt.plot(t_arr, uv_av, lw=1.5, color='red', ls=':', label=r'$<v_x v_y>$')
        plt.plot(t_arr, uw_av, lw=1.5, color='blue', ls=':', label=r'$<v_x v_z>$')
        plt.plot(t_arr, vw_av, lw=1.5, color='green', ls=':', label=r'$<v_y v_z>$')

        plt.legend(fontsize=legend_fsize)
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('t, с', fontsize=label_fsize, fontweight='bold')
        plt.xlim(0, num_ts*ts)
        if ylim:
            plt.ylim(*ylim)
        plt.grid()
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_divergence_field_2d(
            self, figzie=(7, 7), num_levels=20, vmin=-300, vmax=300, grid=True, title_fsize=18,
            fname=None
    ):
        x = self.generator.block.mesh[0][:, :, 0]
        y = self.generator.block.mesh[1][:, :, 0]
        vel = self.generator.get_velocity_field(0)
        div = self.generator.get_divergence(vel, self.generator.block.mesh, self.generator.block.shape)
        div_2d = div[:, :, 0]

        plt.figure(figsize=figzie)
        plt.axes([0.05, 0.05, 0.9, 0.9])
        plt.contourf(x, y, div_2d, num_levels, cmap='rainbow', vmin=vmin, vmax=vmax)
        plt.title(r'$div(U)$', fontsize=title_fsize, fontweight='bold')
        plt.colorbar()
        plt.xticks(x[:, 0], [])
        plt.yticks(y[0, :], [])
        plt.grid(grid)
        if fname:
            plt.savefig(fname)
        plt.show()

    def save_divergence_field_tec(self, fname):
        pass

    def plot_two_point_space_correlation(
            self, i0: int, j0: int, k0: int, ts: float, num_ts: int,
            di: int=1, dj: int=1, dk: int=1, num: int=20,
            figsize=(7, 7), label_fsize=14, ticks_fsize=12, legend_fsize=14, ylim=(-1.1, 1.1),
            axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
        t_arr = np.arange(0, ts * num_ts, ts)
        r = np.zeros(num)
        cor_uu = np.zeros(num)
        cor_vv = np.zeros(num)
        cor_ww = np.zeros(num)
        self.generator.set_puls_node(i0, j0, k0)
        self.generator.time_arr_puls = t_arr
        self.generator.compute_aux_data_time_puls()
        self.generator.compute_pulsation_at_node()
        u0, v0, w0 = self.generator.get_pulsation_at_node()
        u0u0_av = self._get_average(u0 * u0)
        v0v0_av = self._get_average(v0 * v0)
        w0w0_av = self._get_average(w0 * w0)

        for n in range(0, num):
            i = i0 + di * n
            j = j0 + dj * n
            k = k0 + dk * n
            self.generator.set_puls_node(i, j, k)
            self.generator.compute_pulsation_at_node()
            u, v, w = self.generator.get_pulsation_at_node()
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
        if axes:
            plt.axes(axes)
        plt.plot(r, cor_uu, color='red', lw=1.5, label=r'$R_{xx}^r$')
        plt.plot(r, cor_vv, color='blue', lw=1.5, label=r'$R_{yy}^r$')
        plt.plot(r, cor_ww, color='green', lw=1.5, label=r'$R_{zz}^r$')

        plt.grid()
        plt.xlim(xmin=0, xmax=r.max())
        if ylim:
            plt.ylim(*ylim)
        plt.xlabel('r, м', fontsize=label_fsize, fontweight='bold')
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        plt.legend(fontsize=legend_fsize)
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_two_point_time_correlation(
            self, i: int, j: int, k: int, t1: float, t0: float=0.,
            num_dt_av: int=200, num_dt: int=100, figsize=(6.5, 4.5),
            label_fsize=14, ticks_fsize=12, legend_fsize=14, ylim=(-1.1, 1.1),
            axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
        """
        :param num_dt - число отрезков между моментами t1 и t2.
        :param num_dt_av - число отрезков между моментами t0 и t1 и оно же - половина числа
            отрезков ни интервале осреднения.

        1. Момент t2 определяется как t1 + num_dt * (t1 - t0) / num_dt_av.
        2. Момент t3 определяется как t2 + (t1 - t0).
        3. Интервал [t2, t3] при этом разбивается на num_dt_av отрезков.
        4. Пульсации считаются во всех точках между моментами t0 и t3.
        5. Интервал осреднения T = t1 - t0, шаг dt = (t1 - t0) / num_dt_av.
        6. Осреднение проводится от момента t1 - T + i * num_dt_av до момента
            t1 + T + i * num_dt_av, где i = 0, 1, ... num_dt.
        """
        t2 = t1 + num_dt * (t1 - t0) / num_dt_av
        t3 = t2 + t1 - t0
        t_arr = np.linspace(t0, t3, num_dt_av * 2 + num_dt + 1)
        dt_arr = np.linspace(t1, t2, num_dt + 1) - t1
        cor_uu = np.zeros(num_dt + 1)
        cor_vv = np.zeros(num_dt + 1)
        cor_ww = np.zeros(num_dt + 1)
        self.generator.set_puls_node(i, j, k)
        self.generator.time_arr_puls = t_arr
        self.generator.compute_aux_data_time_puls()
        self.generator.compute_pulsation_at_node()
        u, v, w = self.generator.get_pulsation_at_node()
        u0u0_av = self._get_average(u[0: 2 * num_dt_av + 1] * u[0: 2 * num_dt_av + 1])
        v0v0_av = self._get_average(v[0: 2 * num_dt_av + 1] * v[0: 2 * num_dt_av + 1])
        w0w0_av = self._get_average(w[0: 2 * num_dt_av + 1] * w[0: 2 * num_dt_av + 1])
        for n in range(num_dt + 1):
            uu_av = self._get_average(u[n: 2 * num_dt_av + n + 1] * u[n: 2 * num_dt_av + n + 1])
            vv_av = self._get_average(v[n: 2 * num_dt_av + n + 1] * v[n: 2 * num_dt_av + n + 1])
            ww_av = self._get_average(w[n: 2 * num_dt_av + n + 1] * w[n: 2 * num_dt_av + n + 1])
            u0u_av = self._get_average(u[0: 2 * num_dt_av + 1] * u[n: 2 * num_dt_av + n + 1])
            v0v_av = self._get_average(v[0: 2 * num_dt_av + 1] * v[n: 2 * num_dt_av + n + 1])
            w0w_av = self._get_average(w[0: 2 * num_dt_av + 1] * w[n: 2 * num_dt_av + n + 1])
            cor_uu[n] = u0u_av / (np.sqrt(u0u0_av) * np.sqrt(uu_av))
            cor_vv[n] = v0v_av / (np.sqrt(v0v0_av) * np.sqrt(vv_av))
            cor_ww[n] = w0w_av / (np.sqrt(w0w0_av) * np.sqrt(ww_av))

        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(dt_arr, cor_uu, color='red', lw=1.5, label=r'$R_{xx}^t$')
        plt.plot(dt_arr, cor_vv, color='blue', lw=1.5, label=r'$R_{yy}^t$')
        plt.plot(dt_arr, cor_ww, color='green', lw=1.5, label=r'$R_{zz}^t$')

        plt.grid()
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlim(xmin=0, xmax=dt_arr.max())
        if ylim:
            plt.ylim(*ylim)
        plt.xlabel(r't, с', fontsize=label_fsize, fontweight='bold')
        plt.legend(fontsize=legend_fsize)
        if fname:
            plt.savefig(fname)
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

    @classmethod
    def get_spectrum_3d(cls, x: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, num_pnt=100):
        """Вычисление двухмерного энергетического спектра для компоненты скорости на квадратной сетке."""
        u_f = np.fft.fftn(u)
        v_f = np.fft.fftn(v)
        w_f = np.fft.fftn(w)
        u_f = np.fft.fftshift(u_f)
        v_f = np.fft.fftshift(v_f)
        w_f = np.fft.fftshift(w_f)
        length = x.max() - x.min()
        m1, m2, m3 = np.meshgrid(
            np.linspace(-x.shape[0] / 2, x.shape[0] / 2 - 1, x.shape[0]),
            np.linspace(-x.shape[0] / 2, x.shape[0] / 2 - 1, x.shape[0]),
            np.linspace(-x.shape[0] / 2, x.shape[0] / 2 - 1, x.shape[0]),
        )
        m_grid = np.sqrt(m1 ** 2 + m2 ** 2 + m3 ** 2)
        energy = 0.5 * (np.abs(u_f)**2 + np.abs(v_f)**2 + np.abs(w_f)**2) * (1 / x.shape[0])**6
        m_mag = np.linspace(0, m_grid.max(), num_pnt)
        e_k_mag = np.zeros(num_pnt)
        for i in range(num_pnt):
            e_k_mag[i] = cls._get_energy(m_grid, energy, m_mag[i]) * length / (2 * np.pi)
        k_mag = m_mag * 2 * np.pi / length
        return k_mag, e_k_mag

    def plot_spectrum_2d(
            self, figsize=(7, 7), num_pnt=100, ylim=(1e-3, 1e1), xlim=None,
            label_fsize=14, ticks_fsize=12, legend_fsize=14,
            axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
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
        if axes:
            plt.axes(axes)
        plt.plot(k_u, e_u, color='red', label=r'$E_u$', lw=1.5)
        plt.plot(k_v, e_v, color='blue', label=r'$E_v$', lw=1.5)
        plt.plot(k_w, e_w, color='green', label=r'$E_w$', lw=1.5)
        plt.plot(k_w, e_u + e_v + e_w, color='black', label=r'$E_\Sigma$', lw=2.5)

        k = np.logspace(-2, 1, 500)
        if (self.generator.get_desired_spectrum(k) == 0).all():
            pass
        else:
            plt.plot(k, self.generator.get_desired_spectrum(k), color='black', ls='--', lw=1.5, label='Заданный')

        if ylim:
            plt.ylim(*ylim)
        if xlim:
            plt.xlim(*xlim)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(which='both')
        plt.legend(fontsize=legend_fsize)
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('k, 1/м', fontsize=label_fsize, fontweight='bold')
        plt.ylabel('E, м^3/с^2', fontsize=label_fsize, fontweight='bold')
        if fname:
            plt.savefig(fname)
        plt.show()

    def plot_spectrum_3d(
            self, figsize=(7, 7), num_pnt=100, ylim=(1e-3, 1e1), xlim=None,
            label_fsize=14, ticks_fsize=12, legend_fsize=14,
            axes=(0.07, 0.07, 0.9, 0.87), fname=None
    ):
        if self.generator.block.shape[0] != self.generator.block.shape[1] or \
                self.generator.block.shape[0] != self.generator.block.shape[2]:
            raise Exception('Для построения трехмерного спектра блок должен быть кубическим')
        u, v, w = self.generator.get_velocity_field(0)
        x = self.generator.block.mesh[0][:, 0, 0]
        k, e = self.get_spectrum_3d(x, u, v, w, num_pnt)
        plt.figure(figsize=figsize)
        if axes:
            plt.axes(axes)
        plt.plot(k, e, color='red', lw=2.5, label=r'$E_\Sigma$', )

        k = np.logspace(-2, 1, 100)
        if (self.generator.get_desired_spectrum(k) == 0).all():
            pass
        else:
            plt.plot(k, self.generator.get_desired_spectrum(k), color='black', ls='--', lw=1.5, label='Заданный')

        if ylim:
            plt.ylim(*ylim)
        if xlim:
            plt.xlim(*xlim)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(which='both')
        plt.legend(fontsize=legend_fsize)
        plt.xticks(fontsize=ticks_fsize, fontweight='bold')
        plt.yticks(fontsize=ticks_fsize, fontweight='bold')
        plt.xlabel('k, 1/м', fontsize=label_fsize, fontweight='bold')
        plt.ylabel('E, м^3/с^2', fontsize=label_fsize, fontweight='bold')
        if fname:
            plt.savefig(fname)
        plt.show()















