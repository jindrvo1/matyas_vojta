import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


class Frame(np.ndarray):
    def show(self) -> None:
        frame_rgb = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.show()

    @property
    def empty(self) -> bool:
        return self.size == 0

    def __new__(cls, input_array: NDArray = np.array([]), *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(np.ndarray.shape={self.shape})"
