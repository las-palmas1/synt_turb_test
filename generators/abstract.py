import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import enum


class BCType(enum.Enum):
    NotWall = 0
    Wall = 1


class Block:
    """
    Пока предполагается, что в данном классе будет содержаться информация об ортогональном трехмерном блоке,
    включая данные о типе его границ (стенка или не стенка). Сетка может быть неравномерной по всем направлениям.
    Также должны расчитываться шаги сетки по всем направлениям в каждом узле и расстояние до стенки в каждом узле.
    """
    def __init__(self, shape: tuple, mesh: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 bc: List[Tuple[BCType, BCType]]):
        self.shape = shape
        self.mesh = mesh
        self.bc = bc
        self.dim = len(shape)
        self._check_dim()

    def _check_dim(self):
        if len(self.bc) != self.dim:
            raise Exception("Incorrect parameters of block")
        for i in self.mesh:
            if i.shape != self.shape:
                raise Exception("Incorrect parameters of block")


class Generator(metaclass=ABCMeta):
    """
    Базовый класс, в котором должен быть определен метод для расчета поля однородной анизотропной турбулентности
    в заданные моменты времени на 2-D или 3-D сетке. Однородность означает, что параметры турбулентности будут
    постоянны во всех точках области (масштабы, тензор напряжений Рейнольдса и др.),
    а анизотропия будет означать, что матрица тензора напряжений Рейнольдса может быть не только
    пропорцианальной единичной.
    """
    def __init__(
            self, block: Block, u_av: Tuple[float, float, float],
            re_xx: float, re_yy: float, re_zz: float,
            re_xy: float, re_xz: float, re_yz: float,
            time_arr: np.ndarray,
            ):
        """
        :param block: Блок сетки, на которой нужно генерировать пульсации.
        :param u_av: Кортеж из трех значений составляющих осредненной скорости.
        :param re_xx: Осредненное произведение vx*vx.
        :param re_yy: Осредненное произведение vy*vy.
        :param re_zz: Осредненное произведение vz*vz.
        :param re_xy: Осредненное произведение vx*vy.
        :param re_xz: Осредненное произведение vx*vz.
        :param re_yz: Осредненное произведение vy*vz.
        :param time_arr: Моменты времени, для которых производится вычисление поля скоростей.
        """
        self.block = block
        self.u_av = u_av
        self.re_xx = re_xx
        self.re_yy = re_yy
        self.re_zz = re_zz
        self.re_xy = re_xy
        self.re_xz = re_xz
        self.re_yz = re_yz
        self._time_arr_field = time_arr
        self._time_arr_puls = time_arr
        self._i_puls = 0
        self._j_puls = 0
        self._k_puls = 0
        self._vel_field: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._vel_puls: Tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.zeros(time_arr.shape), np.zeros(time_arr.shape), np.zeros(time_arr.shape)
        )
        self._compute_aux_data_common()
        self._compute_aux_data_space()

    def _get_energy_desired(self, k) -> float:
        """Для спектральных методов следует переопределить. По умолчанию возвращает ноль."""
        return 0.

    def get_desired_spectrum(self, k_arr: np.ndarray) -> np.ndarray:
        E_arr = np.array([self._get_energy_desired(k) for k in k_arr])
        return E_arr

    @classmethod
    def get_divergence(cls, vel: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       mesh: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       shape: tuple) -> np.ndarray:
        res = np.zeros(shape)
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                if shape[2] > 1:
                    for k in range(shape[2] - 1):
                        dvx_dx = (vel[0][i + 1, j, k] - vel[0][i, j, k]) / (mesh[0][i + 1, j, k] - mesh[0][i, j, k])
                        dvy_dy = (vel[1][i, j + 1, k] - vel[1][i, j, k]) / (mesh[1][i, j + 1, k] - mesh[1][i, j, k])
                        dvz_dz = (vel[2][i, j, k + 1] - vel[2][i, j, k]) / (mesh[2][i, j, k + 1] - mesh[2][i, j, k])
                        res[i, j, k] = dvx_dx + dvy_dy + dvz_dz
                else:
                    dvx_dx = (vel[0][i + 1, j, 0] - vel[0][i, j, 0]) / (mesh[0][i + 1, j, 0] - mesh[0][i, j, 0])
                    dvy_dy = (vel[1][i, j + 1, 0] - vel[1][i, j, 0]) / (mesh[1][i, j + 1, 0] - mesh[1][i, j, 0])
                    dvz_dz = 0
                    res[i, j, 0] = dvx_dx + dvy_dy + dvz_dz
        res[shape[0]-1, 0: shape[1]-1, 0: shape[2]-1] = res[shape[0]-2, 0: shape[1]-1, 0: shape[2]-1]
        res[0: shape[0], shape[1]-1, 0: shape[2]-1] = res[0: shape[0], shape[1]-2, 0: shape[2]-1]
        if shape[2] > 1:
            res[0: shape[0], 0: shape[1], shape[2]-1] = res[0: shape[0], 0: shape[1], shape[2]-2]
        return res

    @abstractmethod
    def _compute_aux_data_common(self):
        """
        Вычисление вспомогательных данных, общих для всех узлов и независимых от времени.
        """
        pass

    @abstractmethod
    def compute_aux_data_time_puls(self):
        """
        Вычисление вспомогательных данных, общих для всех узлов, но зависимых от времени
        для моментов времени, в которые вычисляются скорости в одном заданном узле.
        """
        pass

    @abstractmethod
    def compute_aux_data_time_field(self):
        """
        Вычисление вспомогательных данных, общих для всех узлов, но зависимых от времени для моментов
        времени, в которое вычисляется все поле скорости.
        """
        pass

    @abstractmethod
    def _compute_aux_data_space(self):
        """
        Вычисление вспомогательных данных, зависимых от координаты, но постоянных во времени
        """
        pass

    def get_velocity_field(self, time_step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает поле скорости на всей сетке на заданном временном шаге."""
        return self._vel_field[time_step]

    def get_pulsation_at_node(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает значение пульсаций в узле в заданные моменты времени."""
        return self._vel_puls

    @abstractmethod
    def compute_velocity_field(self):
        """Расчет поля скорости."""
        pass

    @abstractmethod
    def compute_pulsation_at_node(self):
        """Расчет пульсаций в заданном узле."""
        pass

    def get_puls_node(self):
        return self._i_puls, self._j_puls, self._k_puls

    def set_puls_node(self, i: int, j: int, k: int):
        self._i_puls = i
        self._j_puls = j
        self._k_puls = k

    @property
    def time_arr_field(self):
        return self._time_arr_field

    @time_arr_field.setter
    def time_arr_field(self, value: np.ndarray):
        self._time_arr_field = value

    @property
    def time_arr_puls(self):
        return self._time_arr_puls

    @time_arr_puls.setter
    def time_arr_puls(self, value: np.ndarray):
        self._time_arr_puls = value
