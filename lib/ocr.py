from __future__ import annotations

import functools
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, TypeVar, cast

import easyocr  # type: ignore
import numpy as np
import paddleocr  # type: ignore

from lib.frame import Frame
from lib.logger import Loggable
from lib.utils import OCRResult, suppress_logging

F = TypeVar("F", bound=Callable[..., Any])


def forward_to_ocr(method: F) -> F:
    name = method.__name__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        target = getattr(self.ocr, name)
        return target(*args, **kwargs)

    return cast(F, wrapper)


class OCREngine(Enum):
    EASYOCR = ("easyocr", None)
    PADDLEOCR = ("paddleocr", None)

    def __init__(self, _, cls):
        self._cls = cls

    def create(self, *args, **kwargs):
        return self._cls(*args, **kwargs)


class OCR(Loggable):
    ocr: OCR
    engine: OCREngine
    conf_threshold: float

    def __init__(
        self, engine: OCREngine, confidence_threshold: float = 0.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.conf_threshold = confidence_threshold
        self.ocr = engine.create(
            confidence_threshold=self.conf_threshold, *args, **kwargs
        )

    def _clean_text(self, text: str) -> str:
        return text.strip().upper()

    def process_ocr_result(
        self,
        results: defaultdict[str, list[OCRResult]],
        text: str,
        conf: float,
        points: np.ndarray,
    ) -> defaultdict[str, list[OCRResult]]:
        text = self._clean_text(text)
        conf = float(conf)

        if conf < self.conf_threshold:
            return results

        res_dict: OCRResult = {
            "conf": conf,
            "points": [(int(p[0]), int(p[1])) for p in points],
        }
        results[text].append(res_dict)

        return results

    def process_ocr_run(
        self, all_results: defaultdict[str, list[float]], return_all: bool = False
    ) -> dict[str, float]:
        avgs = self._calc_average_confs(dict(all_results))
        avgs = avgs if return_all else self._select_best_result(avgs)

        return avgs

    def _calc_average_confs(
        self, all_results: dict[str, list[float]]
    ) -> dict[str, float]:
        avgs = {}
        for text, confs in all_results.items():
            avgs[text] = sum(confs) / len(confs)

        return avgs

    def _select_best_result(self, averaged_confs: dict[str, float]) -> dict[str, float]:
        text_best, conf_best = "", 0.0
        for text, conf in averaged_confs.items():
            if conf > conf_best:
                text_best, conf_best = text, conf

        return {text_best: conf_best}

    @forward_to_ocr
    @abstractmethod
    def run_ocr(
        self, frames: list[Frame], return_all: bool = False
    ) -> dict[str, float]: ...

    @forward_to_ocr
    @abstractmethod
    def predict_frame(self, frame: Frame) -> dict[str, list[OCRResult]]: ...

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}({self.ocr.__class__.__name__})(\n"
        res += f"\t'engine': {self.engine}, \n"
        res += f"\t'confidence_threshold': {self.conf_threshold}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")


class PaddleOCR(OCR, Loggable):
    model: paddleocr.PaddleOCR
    conf_threshold: float

    def __init__(self, confidence_threshold: float = 0.5, *args, **kwargs):
        Loggable.__init__(self)
        self.conf_threshold = confidence_threshold
        self.model = self._load_model()

    @suppress_logging
    def _load_model(self) -> paddleocr.PaddleOCR:
        return paddleocr.PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict_frame(self, frame: Frame) -> dict[str, list[OCRResult]]:
        res: defaultdict[str, list[OCRResult]] = defaultdict(list)

        ocr_results = self.model.predict(frame)[0]
        texts = ocr_results["rec_texts"]
        confs = ocr_results["rec_scores"]
        boxes = ocr_results["rec_boxes"]

        for text, conf, box in zip(texts, confs, boxes):
            x1, y1, x2, y2 = box
            res = self.process_ocr_result(
                res, text, conf, np.array([[x1, y1], [x2, y2]])
            )

        return dict(res)

    def run_ocr(
        self, frames: list[Frame], return_all: bool = False
    ) -> dict[str, float]:
        all_results: defaultdict[str, list[float]] = defaultdict(list)
        for frame in frames:
            frame_results = self.predict_frame(frame)
            for text, vals_dict in frame_results.items():
                confs = [v["conf"] for v in vals_dict]
                all_results[text].extend(confs)
        return self.process_ocr_run(all_results, return_all)


class EasyOCR(OCR, Loggable):
    conf_threshold: float

    def __init__(self, confidence_threshold: float = 0.5, *args, **kwargs):
        Loggable.__init__(self)
        self.conf_threshold = confidence_threshold
        self.model = easyocr.Reader(["en"])
        self.allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "

    def predict_frame(self, frame: Frame) -> dict[str, list[OCRResult]]:
        res: defaultdict[str, list[OCRResult]] = defaultdict(list)
        ocr_results = self.model.readtext(np.array(frame))

        for (p1, p2, p3, p4), text, conf in ocr_results:
            res = self.process_ocr_result(
                res, str(text), float(conf), np.array([p1, p2, p3, p4])
            )
        return dict(res)

    def run_ocr(
        self, frames: list[Frame], return_all: bool = False
    ) -> dict[str, float]:
        all_results: defaultdict[str, list[float]] = defaultdict(list)
        for frame in frames:
            frame_results = self.predict_frame(frame)
            for text, vals_dict in frame_results.items():
                confs = [v["conf"] for v in vals_dict]
                all_results[text].extend(confs)

        return self.process_ocr_run(all_results, return_all)


OCREngine.EASYOCR._cls = EasyOCR
OCREngine.PADDLEOCR._cls = PaddleOCR
