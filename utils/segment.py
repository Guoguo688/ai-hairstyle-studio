from __future__ import annotations

import os
import platform
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.image_segmenter import (
    ImageSegmenter,
    ImageSegmenterOptions,
)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_ENV_VAR = "MEDIAPIPE_SELFIE_MODEL_PATH"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_multiclass_256x256.tflite"


def _ensure_model() -> Path:
    env_path = os.getenv(MODEL_ENV_VAR)
    if env_path:
        model_path = Path(env_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"{MODEL_ENV_VAR} points to a missing file: {model_path}"
            )
        return model_path

    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(MODEL_URL, timeout=60)
    response.raise_for_status()
    MODEL_PATH.write_bytes(response.content)
    return MODEL_PATH


def _supports_gpu_delegate() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _create_segmenter() -> ImageSegmenter:
    model_path = _ensure_model()
    base_options = BaseOptions(
        model_asset_path=str(model_path),
        delegate=BaseOptions.Delegate.GPU if _supports_gpu_delegate() else BaseOptions.Delegate.CPU,
    )
    options = ImageSegmenterOptions(
        base_options=base_options,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    try:
        return ImageSegmenter.create_from_options(options)
    except RuntimeError:
        if base_options.delegate != BaseOptions.Delegate.GPU:
            raise
        cpu_options = ImageSegmenterOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=BaseOptions.Delegate.CPU,
            ),
            output_confidence_masks=True,
            output_category_mask=False,
        )
        return ImageSegmenter.create_from_options(cpu_options)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    largest_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_index, 255, 0).astype(np.uint8)


def _detect_face(gray_image: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def _estimate_hair_region(person_mask: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    hair_mask = np.zeros_like(person_mask)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_box = _detect_face(gray_image)

    if face_box is not None:
        x, y, w, h = face_box
        x0 = max(0, x - int(w * 0.35))
        x1 = min(person_mask.shape[1], x + w + int(w * 0.35))
        y0 = max(0, y - int(h * 0.95))
        y1 = min(person_mask.shape[0], y + int(h * 0.28))
        hair_mask[y0:y1, x0:x1] = 255
    else:
        ys, xs = np.where(person_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return hair_mask
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        fallback_height = max(1, int((y1 - y0) * 0.33))
        hair_mask[y0 : y0 + fallback_height, x0:x1] = 255

    hair_mask = cv2.bitwise_and(hair_mask, person_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)
    return hair_mask


def get_hair_mask(image_path: str) -> np.ndarray:
    image_file = Path(image_path).expanduser().resolve()
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {image_file}")

    mp_image = Image.create_from_file(str(image_file))
    with _create_segmenter() as segmenter:
        result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        raise RuntimeError("MediaPipe segmentation did not return a confidence mask.")

    confidence_mask = result.confidence_masks[0].numpy_view()
    if confidence_mask.ndim == 3:
        confidence_mask = confidence_mask[:, :, 0]

    person_mask = (confidence_mask > 0.2).astype(np.uint8) * 255
    person_mask = _largest_component(person_mask)

    hair_mask = _estimate_hair_region(person_mask, image_bgr)
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    _, hair_mask = cv2.threshold(hair_mask, 16, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.GaussianBlur(hair_mask, (31, 31), 0)
    return hair_mask
