from abc import ABC, abstractmethod
import numpy as np


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, daily_data: np.ndarray) -> np.ndarray:
        """
        daily_data: shape (T, F) where T = trading days, F = feature columns
        returns: shape (dim,) float32 vector
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
