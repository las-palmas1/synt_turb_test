import unittest
from analysis_tools import Analyzer
from generators.storage import Lund, Smirnov, OriginalSEM, Davidson
from generators.abstract import Block, BCType
import numpy as np


class LundTest(unittest.TestCase):
    def setUp(self):
        n = 150
        # mesh = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), np.linspace(0, 1, n))
        mesh = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), [0, 1 / (n - 1)])
        self.block = Block(
            shape=(n, n, 2),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.generator = Lund(
            block=self.block,
            u_av=(0., 0., 0.),
            l_t=0.1,
            re_xx=1.,
            re_yy=1.,
            re_zz=1.,
            re_xy=0.,
            re_xz=0.,
            re_yz=0.,
            time_arr=np.array([0])
        )
        self.analyzer = Analyzer(self.generator)

    def test_plot_2d_velocity_field(self):
        self.analyzer.plot_2d_velocity_field(vmin=-2.5, vmax=2.5, grid=False)

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(0, 0, 0, 0.001, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(0, 0, 0, 0.001, 10000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d(grid=False)

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.001, num_ts=1000, di=1, dj=1, dk=0, num=49
        )

    def test_plot_two_point_time_correlation(self):
        self.analyzer.plot_two_point_time_correlation(
            i=0, j=0, k=0, t0=0, t1=1.0, num_dt_av=500, num_dt=500
        )

    def test_plot_spectrum_2d(self):
        self.analyzer.plot_spectrum_2d(num_pnt=200)


class SmirnovTest(unittest.TestCase):
    """
    При единичных временном и линейном масштабе метод Смирнова тождественнен методу Крайхмана. Линейный масштаб
    ни на что не влияет, зачем он нужен - неясно. Двумерный и трехмерный спектры сильно зависит от шага сетки,
    размера области, а также временного масштаба. Полного совпадения с заданным спектром  у двумерного спектра не
    выходит. При некоторых соотношении значений шага, размера области и временного масштаба удается добиться того,
    что к заданному спектру близка правая часть получаемого спектра в плоть до экстремума.
    Однако для трехмерного спектра добиться полного совпадения с заданным возможно.
    Дивергенция уменьшается с уменьшением шага. В пределе вроде бы получается нулевой.
    С графиками истории скорости в точки и графиками вторых моментов вроде бы все нормально. Вышли такими же,
    как и в статье, в которой изложен данный метод.
    """
    def setUp(self):
        n = 70
        size = 20
        # mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), np.linspace(0, size, n))
        mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), [0, size / (n - 1)])
        self.block = Block(
            shape=(n, n, 2),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.generator = Smirnov(
            block=self.block,
            u_av=(0., 0., 0.),
            l_t=1,
            tau_t=1,
            re_xx=1.,
            re_yy=1.,
            re_zz=1.,
            re_xy=0.,
            re_xz=0.,
            re_yz=0.,
            time_arr=np.array([0]),
            mode_num=1000
        )
        self.analyzer = Analyzer(self.generator)

    def test_plot_2d_velocity_field(self):
        self.analyzer.plot_2d_velocity_field(vmin=-2.5, vmax=2.5, grid=False)

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(0, 0, 0, 0.1, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(0, 0, 0, 0.01, 30000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d(vmin=-1.5, vmax=1.5, grid=False)

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.025, num_ts=4000, di=1, dj=1, dk=0, num=49
        )

    def test_plot_two_point_time_correlation(self):
        self.analyzer.plot_two_point_time_correlation(
            i=0, j=0, k=0, t0=0, t1=100, num_dt_av=4000, num_dt=200
        )

    def test_plot_spectrum_2d(self):
        self.analyzer.plot_spectrum_2d(num_pnt=200)

    def test_plot_spectrum_3d(self):
        self.analyzer.plot_spectrum_3d(num_pnt=200)#


class DavidsonTest(unittest.TestCase):
    def setUp(self):
        n = 150
        size = 0.1 * n
        mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), np.linspace(0, size, n))
        # mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), [0, size / (n - 1)])
        self.block = Block(
            shape=(n, n, n),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.generator = Davidson(
            block=self.block,
            u_av=(0., 0., 0.),
            l_t=5,
            tau_t=0.01,
            num_modes=1000,
            viscosity=1.42e-5,
            dissip_rate=1e3,
            k_t=3/2,
            re_xx=1.,
            re_yy=1.,
            re_zz=1.,
            re_xy=0,
            re_xz=0.,
            re_yz=0.,
            time_arr=np.array([0]),
        )
        self.analyzer = Analyzer(self.generator)

    def test_plot_2d_velocity_field(self):
        self.analyzer.plot_2d_velocity_field(vmin=-2.5, vmax=2.5, grid=False)

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(0, 0, 0, 0.001, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(0, 0, 0, 0.001, 5000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d(vmin=-1.5, vmax=1.5, grid=False)

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.002, num_ts=6000, di=1, dj=1, dk=0, num=109
        )

    def test_plot_two_point_time_correlation(self):
        self.analyzer.plot_two_point_time_correlation(
            i=0, j=0, k=0, t0=0, t1=1, num_dt_av=4000, num_dt=500
        )

    def test_plot_spectrum_2d(self):
        self.analyzer.plot_spectrum_2d(num_pnt=200)

    def test_plot_spectrum_3d(self):
        self.analyzer.plot_spectrum_3d(num_pnt=200)


class OriginalSEMTest(unittest.TestCase):
    def setUp(self):
        n = 90
        size = 6.28
        # mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), np.linspace(0, size, n))
        mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), [0, size / (n - 1)])
        self.block = Block(
            shape=(n, n, 2),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.generator = OriginalSEM(
            block=self.block,
            u_av=(10, 0, 0),
            re_xx=1.,
            re_yy=1.,
            re_zz=1.,
            re_xy=0.,
            re_xz=0.,
            re_yz=0.,
            time_arr=np.array([0]),
            sigma=0.5,
            eddy_num=1000
        )
        self.analyzer = Analyzer(self.generator)

    def test_plot_2d_velocity_field(self):
        self.analyzer.plot_2d_velocity_field(figsize=(7, 7), num_levels=20, vmin=-4, vmax=4, grid=False)

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(10, 10, 0, 0.01, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(20, 20, 0, 0.005, 9000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d(vmin=-15, vmax=15, grid=False, num_levels=20)

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.003, num_ts=6000, di=1, dj=1, dk=0, num=25
        )

    def test_plot_two_point_time_correlation(self):
        self.analyzer.plot_two_point_time_correlation(
            i=0, j=0, k=0, t0=0, t1=5, num_dt_av=1000, num_dt=1000)

    def test_plot_spectrum_2d(self):
        self.analyzer.plot_spectrum_2d(num_pnt=200)

    def test_plot_spectrum_3d(self):
        self.analyzer.plot_spectrum_3d(num_pnt=200)


