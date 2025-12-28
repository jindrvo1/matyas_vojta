import cv2
import numpy as np

from lib.frame import Frame


class Contrast:
    @classmethod
    def clahe(
        cls, frame: Frame, clip_limit: float = 3.0, tile_grid_size: int = 8
    ) -> Frame:
        if len(frame.shape) == 2:
            return cls.clahe_gray(frame, clip_limit, tile_grid_size)
        elif len(frame.shape) == 3:
            return cls.clahe_bgr(frame, clip_limit, tile_grid_size)
        else:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

    @staticmethod
    def clahe_bgr(
        frame: Frame, clip_limit: float = 3.0, tile_grid_size: int = 8
    ) -> Frame:
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
        )
        res = np.array(frame)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
        luminence, *colors = cv2.split(res)
        luminence = clahe.apply(luminence)
        lab2 = cv2.merge((luminence, *colors))
        res = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        return Frame(res)

    @staticmethod
    def clahe_gray(
        frame: Frame, clip_limit: float = 3.0, tile_grid_size: int = 8
    ) -> Frame:
        res = np.array(frame)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
        )
        res = clahe.apply(res)

        return Frame(res)

    @staticmethod
    def gamma(frame: Frame, gamma: float = 1.2) -> Frame:
        inv = 1.0 / max(gamma, 1e-6)
        table = ((np.linspace(0, 255, 256) / 255.0) ** inv * 255.0).astype("uint8")
        res = cv2.LUT(frame, table)

        return Frame(res)
