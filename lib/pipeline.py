import pandas as pd
from tqdm import tqdm

from lib.dataset import Dataset
from lib.logger import Loggable
from lib.object_detection import PlaneDetector
from lib.ocr import OCR, OCREngine
from lib.preprocessing import get_processing_func
from lib.video import Video


class Pipeline(Loggable):
    ocr: OCR
    ocr_engine: OCREngine
    plane_detector: PlaneDetector

    def __init__(self, detector: PlaneDetector, ocr: OCR):
        self.detector = detector
        self.ocr = ocr
        self.ocr_engine = ocr.engine

    def process_row(self, row: pd.Series) -> dict:
        video_path = row["Video file"]
        segment_range = row["Segment start"], row["Segment end"]

        video = Video(video_path, target_fps=5, segment_time_range=segment_range)

        cropped = self.detector.detect(
            video.get_frames(),
            confidence_threshold=0.5,
        )
        video.add_frames(cropped, "cropped")

        video.preprocess(
            "cropped", get_processing_func(self.ocr_engine), save_key="preprocessed"
        )

        ocr_res = self.ocr.run_ocr(video.get_frames("preprocessed"))

        return ocr_res

    def process_rows(self, dataset: Dataset) -> Dataset:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            ocr_res = self.process_row(row)
            dataset.add_results(row.name, ocr_res)

        return dataset
