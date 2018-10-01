import numba as nb
import numpy as np


@nb.jit(
    nopython=True
)
def smirnov_compute_velocity_field(
        k1: np.ndarray, k2: np.ndarray, k3: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
        q1: np.ndarray, q2: np.ndarray, q3: np.ndarray,
        omega: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
        a11: np.ndarray, a12: np.ndarray, a13: np.ndarray,
        a21: np.ndarray, a22: np.ndarray, a23: np.ndarray,
        a31: np.ndarray, a32: np.ndarray, a33: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        l_t: float, tau_t: float, mode_num: int, time: float,
):
    shape = x.shape
    v1 = np.zeros(shape)
    v2 = np.zeros(shape)
    v3 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                k1_p = k1 * (l_t / tau_t) / c1[i, j, k]
                k2_p = k2 * (l_t / tau_t) / c2[i, j, k]
                k3_p = k3 * (l_t / tau_t) / c3[i, j, k]
                modes1 = p1 * np.cos(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                ) + q1 * np.sin(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                )
                modes2 = p2 * np.cos(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                ) + q2 * np.sin(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                )
                modes3 = p3 * np.cos(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                ) + q3 * np.sin(
                    (k1_p * x[i, j, k] + k2_p * y[i, j, k] +
                     k3_p * z[i, j, k]) / l_t + omega * time / tau_t
                )
                v1[i, j, k] = np.sqrt(2 / mode_num) * modes1.sum()
                v2[i, j, k] = np.sqrt(2 / mode_num) * modes2.sum()
                v3[i, j, k] = np.sqrt(2 / mode_num) * modes3.sum()
    w1 = c1 * v1
    w2 = c2 * v2
    w3 = c3 * v3
    res = np.zeros((3, shape[0], shape[1], shape[2]))
    res[0] = a11 * w1 + a12 * w2 + a13 * w3
    res[1] = a21 * w1 + a22 * w2 + a23 * w3
    res[2] = a31 * w1 + a32 * w2 + a33 * w3
    return res


