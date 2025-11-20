import cv2
import numpy as np
import easyocr

from lib.frame import Frame
from lib.logger import logger


def preprocess_test(frame: Frame) -> Frame:
    # ...existing code...
    return frame

def enhance_contrast(
    frame: np.ndarray,
    method: str = "clahe",
    clahe_clip: float = 3.0,
    clahe_tile: tuple[int, int] = (8, 8),
    gamma: float = 1.2,
) -> np.ndarray:
    """Robust contrast enhancement. Input can be gray or BGR; returns BGR uint8."""
    if frame is None:
        return frame

    # Ensure numpy array and not empty
    frame = np.asarray(frame)
    if frame.size == 0:
        return frame

    # Ensure uint8
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # If grayscale, convert to BGR for color conversions
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    try:
        if method == "clahe":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
            return out
        if method == "gamma":
            inv = 1.0 / max(gamma, 1e-6)
            table = ((np.linspace(0, 255, 256) / 255.0) ** inv * 255.0).astype("uint8")
            return cv2.LUT(frame, table)
        if method == "hist":
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            ycrcb2 = cv2.merge((y_eq, cr, cb))
            return cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)
    except Exception as e:
        logger.debug(f"enhance_contrast failed: {e}")
    return frame


def run_ocr(
    ocr_reader: easyocr.Reader,
    frames: list[np.ndarray],
    enhance: bool = True,
    method: str = "clahe",
) -> dict[str, list[float]]:
    """
    Apply enhancement robustly per-cropped-frame (handles grayscale/empty),
    convert to RGB and run easyocr.
    """
    detected_registrations = {}
    if not frames:
        return detected_registrations

    for i, frame in enumerate(frames):
        try:
            if frame is None or np.asarray(frame).size == 0:
                continue

            proc = frame.copy()
            if enhance:
                proc = enhance_contrast(proc, method=method)

            # easyocr expects RGB
            if proc.ndim == 3 and proc.shape[2] == 3:
                proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            else:
                # if still single channel, convert to 3-channel RGB-like
                proc_rgb = proc

            ocr_results = ocr_reader.readtext(
                proc_rgb, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
            )
            for _, text, text_conf in ocr_results:
                cleaned = text.strip().upper()
                text_conf = float(text_conf)
                detected_registrations.setdefault(cleaned, []).append(text_conf)
        except Exception as e:
            logger.debug(f"run_ocr: error on frame {i}: {e}")
            continue

    return detected_registrations
# ...existing code...
    return frame


def run_ocr(
    ocr_reader: easyocr.Reader,
    frames: list[np.ndarray],
    enhance: bool = True,
    method: str = "clahe",
) -> dict[str, list[float]]:
    """
    Apply enhancement robustly per-cropped-frame (handles grayscale/empty),
    convert to RGB and run easyocr.
    """
    detected_registrations = {}
    if not frames:
        return detected_registrations

    for i, frame in enumerate(frames):
        try:
            if frame is None or np.asarray(frame).size == 0:
                continue

            proc = frame.copy()
            if enhance:
                proc = enhance_contrast(proc, method=method)

            # easyocr expects RGB
            if proc.ndim == 3 and proc.shape[2] == 3:
                proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            else:
                # if still single channel, convert to 3-channel RGB-like
                proc_rgb = proc

            ocr_results = ocr_reader.readtext(
                proc_rgb, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "
            )
            for _, text, text_conf in ocr_results:
                cleaned = text.strip().upper()
                text_conf = float(text_conf)
                detected_registrations.setdefault(cleaned, []).append(text_conf)
        except Exception as e:
            logger.debug(f"run_ocr: error on frame {i}: {e}")
            continue

    return detected_registrations
# ...existing code...
    return frame




def preprocess_identity(frame: Frame) -> Frame:
    return frame


def preprocess_easyocr(
    frame: Frame,
) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Denoise (optional)
    denoised = cv2.fastNlMeansDenoising(sharpened, h=10)

    # Adaptive thresholding to isolate text
    # thresh = cv2.adaptiveThreshold(
    #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    # )

    # Convert back to 3-channel for saving video
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    return Frame(processed)


def preprocess_paddleocr(frame: Frame) -> Frame:
    # 1️⃣ Denoise (removes compression noise / grass texture)
    # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # 2️⃣ Convert to LAB color space for adaptive contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    el, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(el)
    merged = cv2.merge((cl, a, b))
    frame_cp = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 3️⃣ Slight sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame_cp = cv2.filter2D(frame_cp, -1, kernel)

    # 5️⃣ Convert to RGB (required by PaddleOCR)
    frame_cp = cv2.cvtColor(frame_cp, cv2.COLOR_BGR2RGB)

    return Frame(frame_cp)
