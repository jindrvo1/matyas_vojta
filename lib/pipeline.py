import pandas as pd
from tqdm import tqdm

from lib.dataset import Dataset
from lib.logger import Loggable
from lib.object_detection import PlaneDetector
from lib.ocr import OCR
from lib.postprocessing import Postprocessor
from lib.preprocessing import Preprocessor
from lib.video import VideoFileSource


class Pipeline(Loggable):
    plane_detector: PlaneDetector
    ocr: OCR
    preprocessor: Preprocessor
    postprocessor: Postprocessor | None
    target_fps: int

    def __init__(
        self,
        plane_detector: PlaneDetector,
        ocr: OCR,
        preprocessor: Preprocessor = Preprocessor(),
        postprocessor: Postprocessor | None = None,
        target_fps: int = 5,
    ):
        self.plane_detector = plane_detector
        self.ocr = ocr
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.target_fps = target_fps

    def process_row(
        self, row: pd.Series, target_fps: int | None = None
    ) -> tuple[str, str | None]:
        target_fps = target_fps or self.target_fps
        video_path = row["Video file"]
        segment_range = row["Segment start"], row["Segment end"]

        video = VideoFileSource(
            video_path, target_fps=target_fps, segment_time_range=segment_range
        )

        cropped = self.plane_detector.detect(video.get_frames())
        video.add_frames(cropped, "cropped")

        video.preprocess(
            "cropped", preprocessor=self.preprocessor, save_key="preprocessed"
        )

        ocr_res = self.ocr.run_ocr(video.get_frames(frames_key="preprocessed"))
        ((res_text, _),) = ocr_res.items()

        ocr_res_postprocessed = (
            self.postprocessor.process_registration(res_text)
            if self.postprocessor
            else None
        )

        return res_text, ocr_res_postprocessed

    def process_rows(self, dataset: Dataset) -> Dataset:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            res_text, rest_text_postprocessed = self.process_row(row)

            dataset.add_results(
                row.name,
                ocr_text=res_text,
                ocr_postprocessed=rest_text_postprocessed,
            )

        return dataset

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'plane_detector': {self.plane_detector}, \n"
        res += f"\t'ocr': {self.ocr}, \n"
        res += f"\t'processing_func': {self.preprocessor}, \n"
        res += f"\t'target_fps': {self.target_fps}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")
