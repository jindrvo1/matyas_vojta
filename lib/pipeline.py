import pandas as pd
from tqdm import tqdm

from lib.dataset import Dataset
from lib.logger import Loggable
from lib.object_detection import PlaneDetector
from lib.ocr import OCR
from lib.preprocessing import preprocess_identity
from lib.utils import PreprocessorFn
from lib.video import VideoFileSource


class Pipeline(Loggable):
    plane_detector: PlaneDetector
    ocr: OCR
    processing_func: PreprocessorFn
    target_fps: int

    def __init__(
        self,
        plane_detector: PlaneDetector,
        ocr: OCR,
        processing_func: PreprocessorFn = preprocess_identity,
        target_fps: int = 5,
    ):
        self.plane_detector = plane_detector
        self.ocr = ocr
        self.processing_func = processing_func
        self.target_fps = target_fps

    def process_row(self, row: pd.Series, target_fps: int | None = None) -> dict:
        target_fps = target_fps or self.target_fps
        video_path = row["Video file"]
        segment_range = row["Segment start"], row["Segment end"]

        video = VideoFileSource(
            video_path, target_fps=target_fps, segment_time_range=segment_range
        )

        cropped = self.plane_detector.detect(video.get_frames())
        video.add_frames(cropped, "cropped")

        video.preprocess(
            "cropped", processing_func=self.processing_func, save_key="preprocessed"
        )

        ocr_res = self.ocr.run_ocr(video.get_frames(frames_key="preprocessed"))

        return ocr_res

    def process_rows(self, dataset: Dataset) -> Dataset:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            ocr_res = self.process_row(row)
            dataset.add_results(row.name, ocr_res)

        return dataset

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'plane_detector': {self.plane_detector}, \n"
        res += f"\t'ocr': {self.ocr}, \n"
        res += f"\t'processing_func': {self.processing_func.__name__}, \n"
        res += f"\t'target_fps': {self.target_fps}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")
