import unittest
from analysis_tools import Analyzer
from generators.storage import Lund, Smirnov
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
            u_av=(np.zeros(self.block.shape), np.zeros(self.block.shape), np.zeros(self.block.shape)),
            l_t=0.1,
            re_xx=np.full(self.block.shape, 1),
            re_yy=np.full(self.block.shape, 1),
            re_zz=np.full(self.block.shape, 1),
            re_xy=np.full(self.block.shape, 0),
            re_xz=np.full(self.block.shape, 0),
            re_yz=np.full(self.block.shape, 0),
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
            i=0, j=0, k=0, t0=0, t1=1.0, t2=2, num_dt=40, num_av=500
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
        n = 80
        size = 80
        mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), np.linspace(0, size, n))
        # mesh = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n), [0, size / (n - 1)])
        self.block = Block(
            shape=(n, n, n),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.generator = Smirnov(
            block=self.block,
            u_av=(np.zeros(self.block.shape), np.zeros(self.block.shape), np.zeros(self.block.shape)),
            l_t=1,
            tau_t=1,
            re_xx=np.full(self.block.shape, 1),
            re_yy=np.full(self.block.shape, 1),
            re_zz=np.full(self.block.shape, 1),
            re_xy=np.full(self.block.shape, 0),
            re_xz=np.full(self.block.shape, 0),
            re_yz=np.full(self.block.shape, 0),
            mode_num=1000
        )
        self.analyzer = Analyzer(self.generator)

    def test_plot_2d_velocity_field(self):
        self.analyzer.plot_2d_velocity_field(vmin=-2.5, vmax=2.5, grid=False)

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(0, 0, 0, 0.001, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(0, 0, 0, 0.001, 30000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d(vmin=-1.5, vmax=1.5, grid=False)

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.0005, num_ts=2000, di=1, dj=1, dk=0, num=49
        )

    def test_plot_two_point_time_correlation(self):
        self.analyzer.plot_two_point_time_correlation(
            i=0, j=0, k=0, t0=0, t1=0.1, t2=0.9, num_dt=100, num_av=1000
        )

    def test_plot_spectrum_2d(self):
        self.analyzer.plot_spectrum_2d(num_pnt=200)

    def test_plot_spectrum_3d(self):
        self.analyzer.plot_spectrum_3d(num_pnt=200)

