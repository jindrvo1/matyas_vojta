from __future__ import annotations

import time
from abc import abstractmethod
from typing import Self, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from lib.frame import Frame
from lib.logger import Loggable
from lib.object_detection import PlaneDetector
from lib.ocr import OCR
from lib.preprocessing import preprocess_identity
from lib.utils import BASE_COLORS, OCRResult, Point, PreprocessorFn


class VideoSource(Loggable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_frame(
        self,
        frame: Frame,
        processing_func: PreprocessorFn,
    ) -> Frame:
        return processing_func(frame)

    @abstractmethod
    def play(self, delay_ms: int = 100, inline: bool = True, *args, **kwargs): ...

    @abstractmethod
    def get_frames(
        self, idx_start: int = 0, idx_end: int = -1, *args, **kwargs
    ) -> list[Frame]: ...

    def _load_video(
        self,
        video_path: str,
        target_fps: float | None = None,
        segment_time_range: tuple[int, int] | None = None,
    ) -> list[Frame]:
        cap = cv2.VideoCapture(video_path)

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = target_fps or self.fps

        start_frame, end_frame = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if segment_time_range:
            start_frame, end_frame = self._get_segment_range_frames(
                segment_time_range, self.fps, end_frame
            )

        frames = self._load_frames(cap, start_frame, end_frame)

        return frames

    def _get_segment_range_frames(
        self, segment_time_range: tuple[int, int], fps: float, n_frames: int
    ) -> tuple[int, int]:
        start_sec, end_sec = segment_time_range
        start_frame = max(int(start_sec * fps), 0)
        end_frame = min(int(end_sec * fps), n_frames)

        return start_frame, end_frame

    def _load_frames(
        self,
        video_cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
    ) -> list[Frame]:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        skip = int(self.fps) // int(self.target_fps) if self.target_fps else 1
        frames = []
        while (
            video_cap.isOpened()
            and (frame_idx := int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))) <= end_frame
        ):
            ret, frame = video_cap.read()

            if not ret:
                break

            if frame_idx % skip == 0:
                frames.append(Frame(frame))
        video_cap.release()

        return frames

    def _play(self, frames: list[Frame], delay_ms: int, inline: bool = True):
        if inline:
            self._play_frames_ipython(frames, delay_ms)
        else:
            self._play_frames_cv2(frames, delay_ms)

    def _play_frames_cv2(
        self, frames: list[Frame], delay_ms: int, window_name: str = "Video"
    ):
        window_name = "Video"
        for frame in frames:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

        cv2.destroyWindow(window_name)

    def _play_frames_ipython(self, frames: list[Frame], delay_ms: int):
        delay_s = delay_ms / 1000
        for i, frame in enumerate(frames):
            if frame.shape[-1] == 3:
                clear_output(wait=True)
                plt.imshow(frame[..., ::-1])
                plt.axis("off")
                plt.title(f"Frame {i + 1}/{len(frames)}")
                display(plt.gcf())
                plt.close()
                time.sleep(delay_s)


class CameraSource(VideoSource, Loggable):
    file_path: str
    frames: list[Frame]
    fps: float

    def __init__(self, video_path: str):
        Loggable.__init__(self)
        self.file_path = video_path
        self.frames = self._load_video(video_path)
        self.logger.debug(f"Loaded {len(self.frames)} frames.")

    def play(self, delay_ms: int = 100, inline: bool = True):
        self._play(self.frames, delay_ms, inline)

    def get_frames(self, idx_start: int = 0, idx_end: int = -1) -> list[Frame]:
        return self.frames[idx_start:idx_end]

    def _pad_list_with_none(self, lst: Sequence[int | None], length: int) -> list:
        lst = list(lst)
        n_nones = length - len(lst)
        return [*lst, *([None] * n_nones)]

    def _parse_start_end_frame(
        self,
        start_end_frame: Sequence[int | None],
        start_end_time: Sequence[int | None],
    ) -> tuple[int, int]:
        end_idx = len(self.frames) - 1

        start_frame_res, end_frame_res = 0, end_idx
        start_time, end_time = self._pad_list_with_none(start_end_time, 2)
        start_frame, end_frame = self._pad_list_with_none(start_end_frame, 2)

        start_frame_res = (
            max(0, int(start_time * self.fps)) if start_time else start_frame_res
        )
        end_frame_res = (
            min(end_idx, int(end_time * self.fps)) if end_time else end_frame_res
        )

        start_frame_res = max(0, start_frame) if start_frame else start_frame_res
        end_frame_res = min(end_idx, end_frame) if end_frame else end_frame_res

        return start_frame_res, end_frame_res

    def start_stream_notebook(
        self,
        plane_detector: PlaneDetector,
        ocr: OCR,
        target_fps: int,
        preprocessing_func: PreprocessorFn = preprocess_identity,
        start_end_frame: Sequence[int | None] = (None, None),
        start_end_time: Sequence[int | None] = (None, None),
        pause_on_detection: bool = True,
    ) -> None:
        start_frame, end_frame = self._parse_start_end_frame(
            start_end_frame, start_end_time
        )
        skip = max(int(self.fps // target_fps), 1)
        n_frames = len(self.frames)

        for idx, _ in enumerate(self.frames[start_frame:end_frame], start_frame):
            frame = self.frames[idx]
            self._show_frame(frame, idx, n_frames)

            if idx % skip != 0:
                continue

            cropped_frames = plane_detector.detect_frame(frame)

            if len(cropped_frames) == 0:
                continue

            frames_data = []
            for cropped_frame in cropped_frames:
                preprocessed = preprocessing_func(cropped_frame)
                ocr_results = ocr.predict_frame(preprocessed)
                frames_data.append((cropped_frame, preprocessed, ocr_results))

            self._show_frames_of_interest_boxes(frames_data, idx)
            if pause_on_detection:
                if (
                    input("Press any key to continue, type 'q' to stop... ")
                    .strip()
                    .lower()
                ) == "q":
                    break

    def _show_frame(self, frame: Frame, idx: int, n_frames: int):
        clear_output(wait=True)

        plt.figure(figsize=(6, 4))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb)
        plt.title(f"Frame {idx + 1}/{n_frames}")
        plt.axis("off")
        display(plt.gcf())
        plt.close()

    def _show_frames_of_interest_boxes(
        self,
        frames_data: list[tuple[Frame, Frame, dict[str, list[OCRResult]]]],
        idx: int,
    ):
        if not frames_data:
            return

        clear_output(wait=True)

        all_texts: list[str] = []
        for _, _, ocr_results in frames_data:
            for text in ocr_results.keys():
                if text not in all_texts:
                    all_texts.append(text)

        label_to_color = {
            text: BASE_COLORS[i % len(BASE_COLORS)] for i, text in enumerate(all_texts)
        }

        summary_lines: list[str] = []
        summary_colors: list[tuple[float, ...]] = []

        n_planes = len(frames_data)
        ncols = 2
        nrows = n_planes

        # Height grows with number of planes
        fig_height = 4 * nrows  # tweakable
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, fig_height))

        # If there's only one plane, axes is a 1D array, normalize it to 2D
        if nrows == 1:
            axes = np.array([axes])  # shape (1, 2)

        for plane_idx, (cropped, preprocessed, ocr_results) in enumerate(frames_data):
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pre_rgb = cv2.cvtColor(preprocessed.copy(), cv2.COLOR_BGR2RGB)

            ax_cropped = axes[plane_idx, 0]
            ax_pre = axes[plane_idx, 1]

            # Show cropped
            ax_cropped.imshow(cropped_rgb)
            ax_cropped.set_title(f"Frame {idx + 1} – plane #{plane_idx + 1} (cropped)")
            ax_cropped.axis("off")

            # Draw OCR boxes on preprocessed
            for text, results in ocr_results.items():
                color_rgb_cv2 = tuple(label_to_color[text][0:3])
                color_rgb_matplotlib = tuple(c / 255 for c in color_rgb_cv2)

                for r in results:
                    conf, pts = r["conf"], r["points"]

                    # Collect summary lines with plane index, to make it clear
                    summary_lines.append(
                        f"Plane #{plane_idx + 1} – {text}: {float(conf):.3f}"
                    )
                    summary_colors.append(color_rgb_matplotlib)

                    if len(pts) == 2:
                        self._draw_boxes_paddle(pre_rgb, pts, text, color_rgb_cv2)
                    elif len(pts) == 4:
                        self._draw_boxes_easyocr(pre_rgb, pts, text, color_rgb_cv2)

            ax_pre.imshow(pre_rgb)
            ax_pre.set_title(
                f"Frame {idx + 1} – plane #{plane_idx + 1} (preprocessed + OCR boxes)"
            )
            ax_pre.axis("off")

        # Dynamic space at bottom for the summary text
        extra_bottom = min(0.02 * max(len(summary_lines), 1), 0.18)
        plt.tight_layout(rect=(0, extra_bottom, 1, 1))

        # Draw the text summary under the figure
        if summary_lines:
            y_base = extra_bottom - 0.01
            line_spacing = 0.021  # slightly tighter spacing if many lines

            for i, (line, color) in enumerate(zip(summary_lines, summary_colors)):
                fig.text(
                    0.01, y_base + i * line_spacing, line, color=color, fontsize=10
                )

        display(fig)
        plt.close(fig)

        # Console logging
        print(f"Frame {idx + 1}: {n_planes} plane(s) detected.")
        if any(fd[2] for fd in frames_data):
            print("OCR results:")
            for plane_idx, (_cropped, _pre, ocr_results) in enumerate(frames_data):
                if not ocr_results:
                    print(f"\tPlane #{plane_idx + 1}: <no OCR results>")
                    continue

                print(f"\tPlane #{plane_idx + 1}:")
                for text, results in ocr_results.items():
                    for r in results:
                        print(f"\t\t{text}: {float(r['conf']):.3f}")
        else:
            print("OCR results: <none>")

    def _draw_boxes_paddle(
        self,
        frame: np.ndarray,
        pts: list[Point],
        text: str,
        color: tuple[int, ...],
    ):
        (x1, y1), (x2, y2) = pts
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # bigger text
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_boxes_easyocr(
        self,
        frame: np.ndarray,
        pts: list[Point],
        text: str,
        color: tuple[int, ...],
    ):
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(
            frame,
            [poly],
            isClosed=True,
            color=color,
            thickness=2,
        )

        x1, y1 = poly[0]
        cv2.putText(
            frame,
            text,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'file_path': {self.file_path}, \n"
        res += f"\t'fps': {self.fps}, \n"
        res += f"\t'frames': {len(self.frames)}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")


class VideoFileSource(VideoSource, Loggable):
    file_path: str
    frames: list[Frame]
    frames_dict: dict[str, list[Frame]]
    target_fps: float
    fps: float
    segment_time_range: tuple[int, int] | None

    def __init__(
        self,
        video_path: str,
        target_fps: float | None = None,
        segment_time_range: tuple[int, int] | None = None,
    ):
        self.file_path = video_path
        self.segment_time_range = segment_time_range
        self.frames = self._load_video(video_path, target_fps, segment_time_range)
        self.frames_dict = {"frames": self.frames}

    def get_frames(
        self, idx_start: int = 0, idx_end: int = -1, frames_key: str = "frames"
    ) -> list[Frame]:
        if frames_key not in self.frames_dict:
            raise ValueError(f"Unknown frames key: {frames_key}")
        return self.frames_dict[frames_key][idx_start:idx_end]

    def add_frames(self, frames: list[Frame], frames_key: str) -> Self:
        self.frames_dict[frames_key] = frames
        self.frames = frames
        return self

    def play(
        self, delay_ms: int = 100, inline: bool = True, frames_key: str = "frames"
    ):
        frames = self.get_frames(frames_key=frames_key)
        self._play(frames, delay_ms, inline)

    def preprocess(
        self,
        frames: list[Frame] | str,
        processing_func: PreprocessorFn,
        save_key: str,
    ) -> Self:
        if isinstance(frames, str):
            frames = self.get_frames(frames_key=frames)

        processed_frames = []
        for frame in frames:
            processed_frame = self.preprocess_frame(frame, processing_func)
            processed_frames.append(processed_frame)

        self.add_frames(processed_frames, save_key)

        return self

    def __repr__(self) -> str:
        segment_time_range_str = "<none>"
        if self.segment_time_range is not None:
            start, end = self.segment_time_range[0], self.segment_time_range[1]
            segment_time_range_str = f"00:{start:02d}-00:{end:02d}"

        res = f"{self.__class__.__name__}(\n"
        res += f"\t'file_path': {self.file_path}, \n"
        res += f"\t'fps': {self.fps}, \n"
        res += f"\t'target_fps': {self.target_fps}, \n"
        res += f"\t'segment_time_range': {segment_time_range_str}, \n"
        for i, (key, frames) in enumerate(self.frames_dict.items()):
            res += f"\t'{key}': {len(frames)}{', ' if i < len(self.frames_dict) - 1 else ''}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")
