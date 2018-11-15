import numba as nb
import numpy as np
from typing import List, Tuple


@nb.jit
def davidson_compute_velocity_field(
        k1: np.ndarray, k2: np.ndarray, k3: np.ndarray,
        sigma1: np.ndarray, sigma2: np.ndarray, sigma3: np.ndarray, psi: np.ndarray,
        c1: float, c2: float, c3: float,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
        x: np.ndarray, y: np.ndarray, z: np.ndarray, u_abs: np.ndarray,
        a: float, b: float,
        v1_previous: np.ndarray, v2_previous: np.ndarray, v3_previous: np.ndarray
):
    shape = x.shape
    v1 = np.zeros(shape)
    v2 = np.zeros(shape)
    v3 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                v1[i, j, k] = 2 * (
                        u_abs * np.cos(k1 * x[i, j, k] + k2 * y[i, j, k] + k3 * z[i, j, k] + psi) * sigma1
                ).sum()
                v2[i, j, k] = 2 * (
                        u_abs * np.cos(k1 * x[i, j, k] + k2 * y[i, j, k] + k3 * z[i, j, k] + psi) * sigma2
                ).sum()
                v3[i, j, k] = 2 * (
                        u_abs * np.cos(k1 * x[i, j, k] + k2 * y[i, j, k] + k3 * z[i, j, k] + psi) * sigma3
                ).sum()
    w1 = c1 * v1
    w2 = c2 * v2
    w3 = c3 * v3
    v1_prime = a11 * w1 + a12 * w2 + a13 * w3
    v2_prime = a21 * w1 + a22 * w2 + a23 * w3
    v3_prime = a31 * w1 + a32 * w2 + a33 * w3
    res = np.zeros((3, shape[0], shape[1], shape[2]))
    # для первого шага v1_previous, v2_previous и v3_previous вероятно следует приравнять нулю, а b = 1
    res[0] = a * v1_previous + b * v1_prime
    res[1] = a * v2_previous + b * v2_prime
    res[2] = a * v3_previous + b * v3_prime
    return res


@nb.jit
def davidson_compute_velocity_pulsation(
        k1: List[np.ndarray], k2: List[np.ndarray], k3: List[np.ndarray],
        sigma1: List[np.ndarray], sigma2: List[np.ndarray], sigma3: List[np.ndarray], psi: List[np.ndarray],
        c1: float, c2: float, c3: float,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
        x: float, y: float, z: float, u_abs: np.ndarray,
        a: float, b: float,
        time: np.ndarray
):
    v1 = np.zeros(time.shape)
    v2 = np.zeros(time.shape)
    v3 = np.zeros(time.shape)
    res = np.zeros((3, time.shape[0]))
    for n, t in enumerate(time):
        v1[n] = 2 * (
                u_abs * np.cos(k1[n] * x + k2[n] * y + k3[n] * z + psi[n]) * sigma1[n]
        ).sum()
        v2[n] = 2 * (
                u_abs * np.cos(k1[n] * x + k2[n] * y + k3[n] * z + psi[n]) * sigma2[n]
        ).sum()
        v3[n] = 2 * (
                u_abs * np.cos(k1[n] * x + k2[n] * y + k3[n] * z + psi[n]) * sigma3[n]
        ).sum()
        w1 = c1 * v1[n]
        w2 = c2 * v2[n]
        w3 = c3 * v3[n]
        v1_prime = a11 * w1 + a12 * w2 + a13 * w3
        v2_prime = a21 * w1 + a22 * w2 + a23 * w3
        v3_prime = a31 * w1 + a32 * w2 + a33 * w3
        if n == 0:
            res[0, n] = v1_prime
            res[1, n] = v2_prime
            res[2, n] = v3_prime
        else:
            res[0, n] = a * res[0, n - 1] + b * v1_prime
            res[1, n] = a * res[1, n - 1] + b * v2_prime
            res[2, n] = a * res[2, n - 1] + b * v3_prime
    return res


@nb.njit(parallel=True)
def smirnov_compute_velocity_field(
        k1: np.ndarray, k2: np.ndarray, k3: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
        q1: np.ndarray, q2: np.ndarray, q3: np.ndarray,
        omega: np.ndarray, c1: float, c2: float, c3: float,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        l_t: float, tau_t: float, mode_num: int, time: float,
):
    shape = x.shape
    v1 = np.zeros(shape)
    v2 = np.zeros(shape)
    v3 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in nb.prange(shape[2]):
                k1_p = k1 * (l_t / tau_t) / c1
                k2_p = k2 * (l_t / tau_t) / c2
                k3_p = k3 * (l_t / tau_t) / c3
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


@nb.njit(parallel=True)
def smirnov_compute_pulsation(
        k1: np.ndarray, k2: np.ndarray, k3: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
        q1: np.ndarray, q2: np.ndarray, q3: np.ndarray,
        omega: np.ndarray, c1: float, c2: float, c3: float,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
        x: float, y: float, z: float,
        l_t: float, tau_t: float, mode_num: int, time: np.ndarray,
):
    v1 = np.zeros(time.shape)
    v2 = np.zeros(time.shape)
    v3 = np.zeros(time.shape)
    for n in nb.prange(time.shape[0]):
        k1_p = k1 * (l_t / tau_t) / c1
        k2_p = k2 * (l_t / tau_t) / c2
        k3_p = k3 * (l_t / tau_t) / c3
        modes1 = p1 * np.cos(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        ) + q1 * np.sin(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        )
        modes2 = p2 * np.cos(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        ) + q2 * np.sin(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        )
        modes3 = p3 * np.cos(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        ) + q3 * np.sin(
            (k1_p * x + k2_p * y + k3_p * z) / l_t + omega * time[n] / tau_t
        )
        v1[n] = np.sqrt(2 / mode_num) * modes1.sum()
        v2[n] = np.sqrt(2 / mode_num) * modes2.sum()
        v3[n] = np.sqrt(2 / mode_num) * modes3.sum()
    w1 = c1 * v1
    w2 = c2 * v2
    w3 = c3 * v3
    res = np.zeros((3, time.shape[0]))
    res[0] = a11 * w1 + a12 * w2 + a13 * w3
    res[1] = a21 * w1 + a22 * w2 + a23 * w3
    res[2] = a31 * w1 + a32 * w2 + a33 * w3
    return res


@nb.jit
def original_sem_compute_velocity_field(
        positions: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray, sigma: float, volume: float, eddy_num: int,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
):
    shape = x.shape
    res = np.zeros((positions.shape[0], 3, shape[0], shape[1], shape[2]))
    for n in range(positions.shape[0]):
        x_e = positions[n, 0, :]
        y_e = positions[n, 1, :]
        z_e = positions[n, 2, :]
        epsilon1 = positions[n, 3, :]
        epsilon2 = positions[n, 4, :]
        epsilon3 = positions[n, 5, :]
        u = np.zeros(shape)
        v = np.zeros(shape)
        w = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    form_f1 = (
                            (np.abs((x[i, j, k] - x_e) / sigma) < 1) *
                            (np.sqrt(1.5) * (1 - np.abs((x[i, j, k] - x_e) / sigma)))
                    )
                    form_f2 = (
                            (np.abs((y[i, j, k] - y_e) / sigma) < 1) *
                            (np.sqrt(1.5) * (1 - np.abs((y[i, j, k] - y_e) / sigma)))
                    )
                    form_f3 = (
                            (np.abs((z[i, j, k] - z_e) / sigma) < 1) *
                            (np.sqrt(1.5) * (1 - np.abs((z[i, j, k] - z_e) / sigma)))
                    )
                    f_sigma = np.sqrt(volume) / sigma ** 1.5 * (form_f1 * form_f2 * form_f3)
                    u1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon1).sum()
                    v1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon2).sum()
                    w1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon3).sum()
                    u[i, j, k] = a11 * u1 + a12 * v1 + a13 * w1
                    v[i, j, k] = a21 * u1 + a22 * v1 + a23 * w1
                    w[i, j, k] = a31 * u1 + a32 * v1 + a33 * w1
        res[n, 0, :, :, :] = u
        res[n, 1, :, :, :] = v
        res[n, 2, :, :, :] = w
        return res


@nb.jit
def original_sem_compute_pulsation(
        positions: np.ndarray,
        x: float, y: float, z: float, sigma: float, volume: float, eddy_num: int,
        a11: float, a12: float, a13: float,
        a21: float, a22: float, a23: float,
        a31: float, a32: float, a33: float,
):
    shape = positions.shape
    u = np.zeros(shape[0])
    v = np.zeros(shape[0])
    w = np.zeros(shape[0])
    for n in range(shape[0]):
        x_e = positions[n, 0, :]
        y_e = positions[n, 1, :]
        z_e = positions[n, 2, :]
        epsilon1 = positions[n, 3, :]
        epsilon2 = positions[n, 4, :]
        epsilon3 = positions[n, 5, :]
        form_f1 = (
                (np.abs((x - x_e) / sigma) < 1) *
                (np.sqrt(1.5) * (1 - np.abs((x - x_e) / sigma)))
        )
        form_f2 = (
                (np.abs((y - y_e) / sigma) < 1) *
                (np.sqrt(1.5) * (1 - np.abs((y - y_e) / sigma)))
        )
        form_f3 = (
                (np.abs((z - z_e) / sigma) < 1) *
                (np.sqrt(1.5) * (1 - np.abs((z - z_e) / sigma)))
        )
        f_sigma = np.sqrt(volume) / sigma ** 1.5 * (form_f1 * form_f2 * form_f3)
        u1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon1).sum()
        v1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon2).sum()
        w1 = 1 / (np.sqrt(eddy_num)) * (f_sigma * epsilon3).sum()
        u[n] = a11 * u1 + a12 * v1 + a13 * w1
        v[n] = a21 * u1 + a22 * v1 + a23 * w1
        w[n] = a31 * u1 + a32 * v1 + a33 * w1
    return u, v, w
