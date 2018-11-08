from generators.storage import OriginalSEM, Block, BCType
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as ax3
import matplotlib.animation as anim
import unittest


class TestSEM(unittest.TestCase):
    def setUp(self):
        n = 150
        mesh = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), np.linspace(0, 1, n))
        self.block = Block(
            shape=(n, n, n),
            mesh=(mesh[1], mesh[0], mesh[2]),
            bc=[(BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall), (BCType.NotWall, BCType.NotWall)]
        )
        self.sem = OriginalSEM(
            block=self.block,
            u_av=(np.full(self.block.shape, 0.5), np.full(self.block.shape, 0.5), np.full(self.block.shape, 0.5)),
            sigma=0.3,
            re_xx=np.full(self.block.shape, 1),
            re_yy=np.full(self.block.shape, 1),
            re_zz=np.full(self.block.shape, 1),
            re_xy=np.full(self.block.shape, 0),
            re_xz=np.full(self.block.shape, 0),
            re_yz=np.full(self.block.shape, 0),
            eddy_num=500
        )
        self.sem._compute_init_eddies_pos()

    def test_plot_eddies_centers_movement(self):
        np.random.seed(19680801)
        fig = plt.figure(figsize=(9, 7))
        ax = ax3.Axes3D(fig)

        num_ts = 100
        eddy_params = list(self.sem.get_eddies_params(4, num_ts))
        ax.plot(xs=[self.sem.x_min, self.sem.x_min], ys=[self.sem.y_min, self.sem.y_min],
                zs=[self.sem.z_min, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_min], ys=[self.sem.y_min, self.sem.y_max],
                zs=[self.sem.z_min, self.sem.z_min], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_max], ys=[self.sem.y_min, self.sem.y_min],
                zs=[self.sem.z_min, self.sem.z_min], c='red', lw=2)
        ax.plot(xs=[self.sem.x_max, self.sem.x_max], ys=[self.sem.y_min, self.sem.y_min],
                zs=[self.sem.z_min, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_max, self.sem.x_max], ys=[self.sem.y_min, self.sem.y_max],
                zs=[self.sem.z_min, self.sem.z_min], c='red', lw=2)
        ax.plot(xs=[self.sem.x_max, self.sem.x_max], ys=[self.sem.y_max, self.sem.y_max],
                zs=[self.sem.z_min, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_max], ys=[self.sem.y_max, self.sem.y_max],
                zs=[self.sem.z_min, self.sem.z_min], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_min], ys=[self.sem.y_max, self.sem.y_max],
                zs=[self.sem.z_min, self.sem.z_max], c='red', lw=2)

        ax.plot(xs=[self.sem.x_min, self.sem.x_max], ys=[self.sem.y_min, self.sem.y_min],
                zs=[self.sem.z_max, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_min], ys=[self.sem.y_min, self.sem.y_max],
                zs=[self.sem.z_max, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_max, self.sem.x_max], ys=[self.sem.y_min, self.sem.y_max],
                zs=[self.sem.z_max, self.sem.z_max], c='red', lw=2)
        ax.plot(xs=[self.sem.x_min, self.sem.x_max], ys=[self.sem.y_max, self.sem.y_max],
                zs=[self.sem.z_max, self.sem.z_max], c='red', lw=2)
        line = ax.plot(eddy_params[0][0], eddy_params[0][1], eddy_params[0][2], ls='', marker='o')[0]

        def update(frame):
            line.set_data([eddy_params[frame][0], eddy_params[frame][1]])
            line.set_3d_properties(eddy_params[frame][2])

        ani = anim.FuncAnimation(fig, func=update, frames=num_ts,
                                 interval=40)
        plt.show()

