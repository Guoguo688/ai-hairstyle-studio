from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.painter import (
    SILICONFLOW_API_KEY,
    PainterError,
    generate_hairstyle_image,
)
from utils.segment import get_hair_mask


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _has_valid_key(value: str) -> bool:
    return bool(value and not value.startswith("YOUR_") and not value.startswith("<"))


def _save_output_image(image: Image.Image) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"hairstyle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(output_path, format="PNG")
    return output_path


def _image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="AI Hairstyle Studio", layout="wide")
    st.title("AI Hairstyle Studio")
    st.caption("上传一张人物照片，输入你想要的发型和颜色，生成新的发型效果图。")

    with st.sidebar:
        st.header("使用说明")
        st.write("1. 上传一张清晰的人像照片。")
        st.write("2. 输入发型和颜色需求，例如：法式慵懒卷发，亚麻棕色。")
        st.write("3. 点击“开始改造”，等待系统完成识别、提示词优化和发型生成。")
        st.write("4. 结果会自动保存到 `outputs/` 文件夹，并可直接下载。")

        missing_keys = []
        if not _has_valid_key(SILICONFLOW_API_KEY):
            missing_keys.append("SILICONFLOW_API_KEY")

        if missing_keys:
            st.warning(
                "请先在 `.env` 中填写这些 Key："
                + "、".join(missing_keys)
                + "。填好后刷新页面即可。"
            )
        else:
            st.success("API Key 已读取，可以开始生成。")

    uploaded_file = st.file_uploader(
        "上传图片",
        type=["png", "jpg", "jpeg", "webp"],
    )
    hairstyle_request = st.text_input(
        "输入发型和颜色",
        placeholder="例如：法式慵懒卷发，亚麻棕色",
    )

    left_col, right_col = st.columns(2)

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        with left_col:
            st.subheader("原图")
            st.image(original_image, use_container_width=True)
    else:
        original_image = None

    start = st.button("开始改造", type="primary", use_container_width=True)

    if start:
        missing_keys = []
        if not _has_valid_key(SILICONFLOW_API_KEY):
            missing_keys.append("SILICONFLOW_API_KEY")

        if missing_keys:
            st.info(
                "还不能开始生成。请先在 `.env` 中填写："
                + "、".join(missing_keys)
                + "。"
            )
            return

        if original_image is None:
            st.error("请先上传图片。")
            return

        if not hairstyle_request.strip():
            st.error("请先输入发型和颜色需求。")
            return

        temp_input_path = PROJECT_ROOT / "_tmp_input.png"
        progress = st.progress(0)
        status = st.empty()

        try:
            original_image.save(temp_input_path, format="PNG")

            status.write("正在识别头发...")
            progress.progress(20)
            hair_mask = get_hair_mask(str(temp_input_path))

            status.write("正在沟通 AI 画师...")
            progress.progress(55)

            status.write("发型生成中...")
            progress.progress(75)
            result_image = generate_hairstyle_image(
                image_path=str(temp_input_path),
                mask=hair_mask,
                user_request_cn=hairstyle_request,
            )

            output_path = _save_output_image(result_image)
            progress.progress(100)
            status.success(f"改造完成，已保存到 {output_path}")

            with right_col:
                st.subheader("生成效果图")
                st.image(result_image, use_container_width=True)
                st.download_button(
                    "下载结果图",
                    data=_image_bytes(result_image),
                    file_name=output_path.name,
                    mime="image/png",
                    use_container_width=True,
                )

        except PainterError as exc:
            progress.empty()
            status.empty()
            st.error(f"生成失败：{exc}")
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"处理失败：{exc}")
        finally:
            if temp_input_path.exists():
                try:
                    temp_input_path.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
