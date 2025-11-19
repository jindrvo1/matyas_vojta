import cv2
import numpy as np

from lib.frame import Frame


def preprocess_test(frame: Frame) -> Frame:
    # Upravy
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
