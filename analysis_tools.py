from generators.abstract import Generator
import matplotlib.pyplot as plt
from typing import Tuple


class Analyzer:
    def __init__(self, generator: Generator):
        self.generator = generator

    def plot_2d_velocity_field(self):
        pass

    def save_velocity_field_tec(self, fname):
        pass

    def plot_velocity_history(self, r: Tuple[float, float, float], ts, num_ts: int):
        pass

    def plot_moments(self, r: Tuple[float, float, float], ts: float, num_ts: int):
        pass

    def plot_divergence_field_2d(self):
        pass

    def save_divergence_field_tec(self, fname):
        pass

    def plot_two_point_space_correlation(self, r: Tuple[float, float, float],
                                         dr: Tuple[float, float, float], ts: float, num_ts: int):
        pass

    def plot_two_point_time_correlation(self, r: Tuple[float, float, float],
                                        dt: float, ts: float, num_ts: int):
        pass


