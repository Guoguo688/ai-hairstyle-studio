from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["all_proxy"] = ""
os.environ["ALL_PROXY"] = ""

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "").strip()

SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
).rstrip("/")

QWEN_IMAGE_EDIT_MODEL = os.getenv(
    "SILICONFLOW_INPAINTING_MODEL",
    "Qwen/Qwen-Image-Edit",
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted face, extra limbs, duplicate features, "
    "bad hairline, unrealistic hair strands, broken anatomy, artifacts"
)


class PainterError(RuntimeError):
    pass


def _require_api_key(name: str, value: str) -> str:
    if value and not value.startswith("YOUR_") and not value.startswith("<"):
        return value
    raise PainterError(f"Missing valid API key: {name}")


def _to_png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _normalize_mask(mask: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(mask, Image.Image):
        mask_image = mask.convert("L")
    else:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
        mask_array = np.clip(mask_array, 0, 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode="L")

    # SiliconFlow inpainting generally expects a strict black-white mask.
    mask_array = np.array(mask_image)
    mask_array = np.where(mask_array > 16, 255, 0).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L")


def _extract_generated_image(response_json: dict[str, Any]) -> str:
    images = response_json.get("images")
    if not images:
        raise PainterError(f"SiliconFlow returned no images: {response_json}")

    first = images[0]
    if "url" in first and first["url"]:
        return first["url"]
    if "b64_json" in first and first["b64_json"]:
        return f"data:image/png;base64,{first['b64_json']}"
    raise PainterError(f"Unsupported SiliconFlow image payload: {first}")


def _decode_image_reference(image_ref: str) -> Image.Image:
    if image_ref.startswith("data:image/"):
        _, encoded = image_ref.split(",", 1)
        binary = base64.b64decode(encoded)
        return Image.open(BytesIO(binary)).convert("RGB")

    response = requests.get(image_ref, timeout=120)
    if not response.ok:
        raise PainterError(
            f"Failed to download generated image: {response.status_code} {response.text}"
        )
    return Image.open(BytesIO(response.content)).convert("RGB")


def generate_inpainted_image(
    image_path: str | os.PathLike[str],
    mask: np.ndarray | Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Image.Image:
    api_key = _require_api_key("SILICONFLOW_API_KEY", SILICONFLOW_API_KEY)

    source_image = Image.open(image_path).convert("RGB")
    mask_image = _normalize_mask(mask).resize(source_image.size, Image.Resampling.LANCZOS)

    payload = {
        "model": QWEN_IMAGE_EDIT_MODEL,
        "prompt": prompt,
        "image": _to_png_data_url(source_image),
        "mask": _to_png_data_url(mask_image),
    }

    response = requests.post(
        f"{SILICONFLOW_BASE_URL}/images/generations",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )

    if not response.ok:
        trace_id = response.headers.get("x-siliconcloud-trace-id", "")
        trace_suffix = f", trace_id={trace_id}" if trace_id else ""
        raise PainterError(
            f"SiliconFlow request failed: {response.status_code} {response.text}{trace_suffix}"
        )

    image_ref = _extract_generated_image(response.json())
    return _decode_image_reference(image_ref)


def generate_hairstyle_image(
    image_path: str | os.PathLike[str],
    mask: np.ndarray | Image.Image,
    user_request_cn: str,
) -> Image.Image:
    return generate_inpainted_image(
        image_path=image_path,
        mask=mask,
        prompt=user_request_cn.strip(),
    )
