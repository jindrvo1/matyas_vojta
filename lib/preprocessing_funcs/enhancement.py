import cv2
import numpy as np

from lib.frame import Frame


class Enhancement:
    @staticmethod
    def sharpen(frame: Frame) -> Frame:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel)

        return Frame(sharpened)

    @classmethod
    def denoise(
        cls,
        frame: Frame,
        h: int = 5,
        hColor: int = 3,
        templateWindowSize: int = 7,
        searchWindowSize: int = 11,
    ) -> Frame:
        if len(frame.shape) == 2:
            return cls.denoise_gray(frame, h, templateWindowSize, searchWindowSize)
        elif len(frame.shape) == 3:
            return cls.denoise_colored(
                frame, h, hColor, templateWindowSize, searchWindowSize
            )
        else:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

    @staticmethod
    def denoise_colored(
        frame: Frame,
        h: int = 5,
        hColor: int = 3,
        templateWindowSize: int = 7,
        searchWindowSize: int = 11,
    ) -> Frame:
        res = cv2.fastNlMeansDenoisingColored(
            frame, None, h, hColor, templateWindowSize, searchWindowSize
        )

        return Frame(res)

    @staticmethod
    def denoise_gray(
        frame: Frame,
        h: int = 5,
        templateWindowSize: int = 7,
        searchWindowSize: int = 11,
    ) -> Frame:
        res = cv2.fastNlMeansDenoising(
            frame, None, h, templateWindowSize, searchWindowSize
        )

        return Frame(res)

    @staticmethod
    def adaptative_threshold(frame: Frame, block_size: int = 11, c: float = 2) -> Frame:
        res = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )

        return Frame(res)

    @staticmethod
    def resize_for_ocr(frame: Frame, min_height: int = 40) -> Frame:
        res = np.array(frame)
        h, w = frame.shape[:2]
        if h < min_height:
            scale = min_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            res = cv2.resize(res, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return Frame(res)
