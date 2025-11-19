from __future__ import annotations

from ultralytics import YOLO  # type: ignore[import-untyped]

from lib.frame import Frame
from lib.logger import Loggable


class PlaneDetector(Loggable):
    model: YOLO
    confidence_threshold: float

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.model = YOLO(model_name, verbose=False)
        self.confidence_threshold = confidence_threshold

    def detect_frame(
        self, frame: Frame, confidence_threshold: float | None = None
    ) -> Frame:
        confidence_threshold = confidence_threshold or self.confidence_threshold
        cropped_frame: Frame = Frame()
        result = self.model(frame, verbose=False)[0]

        if len(result.boxes) == 0:
            self.logger.debug("❌ Frame does not contain any object.")
            return cropped_frame

        for box in result.boxes:
            object_type, conf = int(box.cls[0]), float(box.conf[0])
            label = self.model.names[object_type]
            self.logger.debug(f"Object type: {label}, confidence: {conf:.2f}")

            if object_type != 4:
                self.logger.debug("❌ Frame does not contain an airplane.")
                continue

            if conf < confidence_threshold:
                self.logger.debug(
                    "❌ The confidence of the object being in airplane is too low."
                )
                continue

            x1, y1, x2, y2 = tuple(map(int, box.xyxy[0]))
            crop = Frame(frame[y1:y2, x1:x2])

            airplane_in_frame = self._check_that_airplane_is_in_frame(
                crop, box_aspect_ratio_min=2.0
            )

            self.logger.debug(f"Airplane in frame: {airplane_in_frame}.")
            if not airplane_in_frame:
                continue

            self.logger.debug("✅ Frame contains an airplane.")
            cropped_frame = crop

        return cropped_frame

    def detect(
        self, frames: list[Frame], confidence_threshold: float | None = None
    ) -> list[Frame]:
        self.logger.debug(f"Total frames to process: {len(frames)}.")

        confidence_threshold = confidence_threshold or self.confidence_threshold
        cropped_frames = []
        for frame in frames:
            cropped_frame = self.detect_frame(frame, confidence_threshold)
            if not cropped_frame.empty:
                cropped_frames.append(cropped_frame)

        self.logger.debug(f"Total frames found: {len(cropped_frames)}.")

        return cropped_frames

    def _check_that_airplane_is_in_frame(
        self, cropped_frame: Frame, box_aspect_ratio_min: float = 2.0
    ) -> bool:
        # If the bounding box's aspect ratio is too small, the airplane is probably largely off-screen
        h, w, _ = cropped_frame.shape
        box_aspect_ratio = w / h

        self.logger.debug(
            f"Bounding box aspect ratio={box_aspect_ratio:.2f} (min={box_aspect_ratio_min:.2f})"
        )

        return False if box_aspect_ratio < box_aspect_ratio_min else True

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'model': {self.model.model_name}, \n"
        res += f"\t'confidence_threshold': {self.confidence_threshold}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")
