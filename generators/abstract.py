import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import enum


class BCType(enum.Enum):
    NotWall = 0
    Wall = 1


class Block:
    """
    Пока предполагается, что в данном классе будет содержаться информация о прямоугольном (2d или 3d) блоке,
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
    Базовый класс, в котором должен быть определен метод для расчета поля мгновенной скорости в заданные
    моменты времени на 2-D или 3-D сетке.
    """
    def __init__(self, block: Block, u_av: Tuple[np.ndarray, np.ndarray, np.ndarray], l_t: float,
                 re_xx: np.ndarray, re_yy: np.ndarray, re_zz: np.ndarray,
                 re_xy: np.ndarray, re_xz: np.ndarray, re_yz: np.ndarray, **kwargs):
        """
       :param block: Блок сетки, на которой нужно генерировать пульсации.
        :param u_av: Кортеж из трех массивов составляющих осредненной скорости для каждого узла сетки.
        :param l_t: Линейный масштаб турбулентности.
        :param re_xx: Осредненное произведение vx*vx.
        :param re_yy: Осредненное произведение vy*vy.
        :param re_zz: Осредненное произведение vz*vz.
        :param re_xy: Осредненное произведение vx*vy.
        :param re_xz: Осредненное произведение vx*vz.
        :param re_yz: Осредненное произведение vy*vz.
        :param kwargs:
        """
        self.block = block
        self.u_av = u_av
        self.l_t = l_t
        self.kwargs = kwargs
        self.re_xx = re_xx
        self.re_yy = re_yy
        self.re_zz = re_zz
        self.re_xy = re_xy
        self.re_xz = re_xz
        self.re_yz = re_yz

    @classmethod
    def _get_energy_desired(cls, k) -> float:
        """Для спектральных методов следует переопределить. По умолчанию возвращает ноль."""
        return 0.

    def get_desired_spectrum(self, k1, k2, num: int) -> Tuple[np.ndarray, np.ndarray]:
        k_arr = np.linspace(k1, k2, num)
        E_arr = np.array([self._get_energy_desired(k) for k in k_arr])
        return k_arr, E_arr

    @abstractmethod
    def get_velocity_field(self, time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вчисление поля скорости на всей сетке в заданный момент времени."""
        pass

    @abstractmethod
    def get_pulsation_at_node(self, i: int, j: int, k: int, time: np.ndarray) -> Tuple[float, float, float]:
        """Вычисление пульсаций в узле в заданные моменты времени."""
        pass

    @abstractmethod
    def _compute_aux_data(self):
        """
        Тут следует вычислять вспомогательные данные, чтобы не дублировать их вычиление при вызовах методов
        для расчета поля скорости.
        """
        pass


