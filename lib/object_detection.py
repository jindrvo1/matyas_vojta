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
    ) -> list[Frame]:
        confidence_threshold = confidence_threshold or self.confidence_threshold
        cropped_frames: list[Frame] = []
        result = self.model(frame, verbose=False)[0]

        if len(result.boxes) == 0:
            self.logger.debug("❌ Frame does not contain any object.")
            return cropped_frames

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
            cropped_frames.append(crop)

        return cropped_frames

    def detect(
        self, frames: list[Frame], confidence_threshold: float | None = None
    ) -> list[Frame]:
        self.logger.debug(f"Total frames to process: {len(frames)}.")

        confidence_threshold = confidence_threshold or self.confidence_threshold
        cropped_frames_res, n_cropped_frames, max_simultaneously = [], 0, 0
        for frame in frames:
            cropped_frames = self.detect_frame(frame, confidence_threshold)
            if len(cropped_frames) > 0:
                # This is a temporary solution that, in cases of multiple airplanes in the frame,
                # puts all of the returned frames into one list. Hence, OCR will only yield one
                # result. In the future, this should be changed to return a list of lists
                # and OCR should be able to handle multiple airplanes.
                cropped_frames_res += cropped_frames

                n_cropped_frames += len(cropped_frames)
                max_simultaneously = max(max_simultaneously, len(cropped_frames))

        self.logger.debug(
            f"Total frames found: {n_cropped_frames} (max {max_simultaneously} simultaneously)."
        )

        return cropped_frames_res

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
