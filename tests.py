import unittest
from analysis_tools import Analyzer
from generators.storage import Lund
from generators.abstract import Block, BCType
import numpy as np


class LundTest(unittest.TestCase):
    def setUp(self):
        mesh = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50), [0])
        self.block = Block(
            shape=(50, 50, 1),
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
        self.analyzer.plot_2d_velocity_field()

    def test_plot_velocity_history(self):
        self.analyzer.plot_velocity_history(0, 0, 0, 0.001, 1000)

    def test_plot_moments(self):
        self.analyzer.plot_moments(0, 0, 0, 0.001, 6000)

    def test_plot_divergence_field_2d(self):
        self.analyzer.plot_divergence_field_2d()

    def test_plot_two_point_space_correlation(self):
        self.analyzer.plot_two_point_space_correlation(
            i0=0, j0=0, k0=0, ts=0.001, num_ts=1000, di=1, dj=1, dk=0, num=49
        )

