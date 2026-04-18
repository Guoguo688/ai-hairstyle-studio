# AI 虚拟理发工作室

一个面向发型预览场景的本地化 AI 应用：用户上传人物照片，输入中文发型需求，系统先在本地完成头发区域抠图，再调用 Qwen-Image-Edit 做局部重绘，生成新的发型效果图。

## 项目简介

`AI 虚拟理发工作室` 使用 `Streamlit` 构建交互界面，结合本地视觉分割与云端图像编辑能力，实现较轻量、可落地的发型改造流程。

当前代码状态下，核心流程如下：

1. 用户上传头像或人物照片。
2. 本地使用 `MediaPipe` 模型识别人像区域，并估算头发遮罩。
3. 将原图、遮罩和中文发型描述发送到 `SiliconFlow` 的图像编辑接口。
4. 由 `Qwen/Qwen-Image-Edit` 完成局部重绘，仅修改头发区域。
5. 结果图自动保存到 `outputs/` 目录，并支持页面内下载。

## 技术方案

### 1. MediaPipe 本地抠图（CPU 模式）

项目在 `utils/segment.py` 中使用本地 `selfie_multiclass_256x256.tflite` 模型执行图像分割，并显式设置：

```python
delegate=BaseOptions.Delegate.CPU
```

这意味着当前方案不依赖 GPU，也不需要额外的远端分割服务，适合本地快速启动和稳定调试。

实现特点：

- 使用 `MediaPipe ImageSegmenter` 生成人像置信度遮罩。
- 保留最大连通区域，减少背景误检。
- 结合 `OpenCV` 人脸检测估算头发区域。
- 通过闭运算、开运算和高斯模糊优化遮罩边缘，提升局部重绘质量。

### 2. Qwen-Image-Edit 局部重绘

项目在 `utils/painter.py` 中通过 `SiliconFlow` 接入：

- 模型：`Qwen/Qwen-Image-Edit`
- 能力：基于原图 + mask 的局部重绘
- 调用方式：向 `SiliconFlow` 图像生成接口提交 `image`、`mask` 和中文 prompt

当前实现会：

- 将原图转为 PNG Data URL
- 将遮罩标准化为严格黑白掩码
- 调用 `/images/generations` 接口完成编辑
- 自动处理返回的图片 URL 或 Base64 结果

## 避坑指南

这个项目目前有三个很关键、也很有代表性的落地亮点。

### 1. 代理屏蔽

在部分本地环境中，请求会被系统代理或环境代理拦截，导致访问 `SiliconFlow` 出现异常、超时或不可预期的网络问题。

当前代码已在 `utils/painter.py` 中主动清空以下环境变量：

- `http_proxy`
- `https_proxy`
- `HTTP_PROXY`
- `HTTPS_PROXY`
- `all_proxy`
- `ALL_PROXY`

这样可以避免请求误走代理，减少“明明本机能联网，但接口始终失败”的隐性问题。

### 2. URL 404 修复

图像编辑接口如果路径写错，很容易直接返回 `404`。

当前代码已修正为：

```text
{SILICONFLOW_BASE_URL}/images/generations
```

这一步很关键。它保证了图像编辑请求命中正确的服务地址，而不是因为路径错误导致整条生成链路中断。

### 3. 移除翻译逻辑，直接使用中文

当前实现已经不再增加“中文先翻译成英文再生成”的中间步骤，而是直接把用户输入的中文发型需求传给图像编辑模型。

这带来三个直接收益：

- 减少额外依赖和中间失败点
- 避免翻译偏差影响发型描述
- 让产品交互更自然，用户直接输入中文即可

在当前代码中，`generate_hairstyle_image(...)` 会直接使用 `user_request_cn` 作为 prompt 提交给模型，这也是项目体验明显变顺的一步。

## 环境要求

建议使用本地 Python 虚拟环境，并确保安装了项目运行所需依赖。根据当前代码，至少需要以下能力对应的库：

- `streamlit`
- `Pillow`
- `opencv-python`
- `mediapipe`
- `python-dotenv`
- `requests`
- `numpy`

此外，项目根目录下需要准备 `.env` 文件，并至少配置：

```env
SILICONFLOW_API_KEY=你的_SiliconFlow_API_Key
```

可选配置项包括：

```env
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_INPAINTING_MODEL=Qwen/Qwen-Image-Edit
```

同时，请确认以下本地模型文件存在：

```text
utils/models/selfie_multiclass_256x256.tflite
```

如果该文件缺失，本地头发区域抠图将无法执行。

## 启动说明

在项目根目录执行：

```bash
streamlit run app.py
```

启动后，浏览器会打开应用页面。基本使用流程如下：

1. 上传一张清晰的人物照片。
2. 输入中文发型需求，例如“法式慵懒卷发，亚麻棕色”。
3. 点击开始生成。
4. 等待系统完成头发识别、局部重绘与结果保存。
5. 在页面中查看效果图，并从 `outputs/` 目录获取生成结果。

## 项目价值

这个项目的完成度，体现在它不是停留在“模型能调用”的演示阶段，而是已经打通了一个相对完整的业务闭环：

- 有可直接运行的 `Streamlit` 前端入口
- 有本地 CPU 分割能力，降低部署门槛
- 有稳定的图像编辑接口接入
- 有真实踩坑后的工程修复方案
- 有中文直输的用户体验优化

它已经具备一个“AI 发型预览小工作室”的雏形，既能展示 AI 图像编辑能力，也体现了工程侧把功能真正跑通的价值。
