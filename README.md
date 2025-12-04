# Z-Image-Turbo on Apple Silicon

> Run Z-Image-Turbo locally on your Mac with just two commands.

[English](#features) | [中文](#功能特性)

**Z-Image-Turbo** is an efficient single-stream diffusion transformer model from [Tongyi Lab](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo). This project provides a **plug-and-play** solution to run it natively on Apple Silicon (M1/M2/M3/M4) with an intuitive web interface.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Your Mac (M1/M2/M3/M4)                                    │
│                                                             │
│   ┌─────────────┐      ┌─────────────────────────────────┐  │
│   │   Gradio    │ HTTP │     FastAPI Model Server        │  │
│   │   Web UI    │─────▶│   Z-Image-Turbo (MPS)           │  │
│   │  (main.py)  │      │   (model_server.py)             │  │
│   └─────────────┘      └─────────────────────────────────┘  │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                           │
│   │  Grok API   │  Optional: AI-powered prompt enhancement  │
│   │  (xAI)      │                                           │
│   └─────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Native Apple Silicon Support** - Optimized for MPS (Metal Performance Shaders)
- **Fast Inference** - ~8 steps for high-quality images with bfloat16 acceleration
- **Two Input Modes**
  - Chat Mode: Describe in natural language, AI enhances your prompt
  - Direct Mode: Full control with your own detailed prompts
- **Flexible Resolutions** - 512px to 1280px with various aspect ratios
- **History & Persistence** - All generated images saved to local database
- **Bilingual UI** - English and Chinese interface

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/OrdinarySF/z-image-inference.git
cd z-image-inference

# Install dependencies (uv will handle everything)
uv sync
```

### Run

**Step 1: Start the model server** (first run will download ~6GB model)

```bash
uv run python model_server.py
```

**Step 2: In a new terminal, launch the web UI**

```bash
uv run python main.py
```

Open http://127.0.0.1:7860 in your browser. That's it!

## Usage

### Direct Input Mode

Perfect for users who want full control over the prompt:

1. Go to **"Direct Input Prompt"** tab
2. Enter your detailed prompt
3. Adjust resolution and steps
4. Click **"Generate Image"**

### Chat Mode (Optional)

Requires [xAI API key](https://x.ai/) for AI-powered prompt enhancement:

1. Create `.env` file with your API key:
   ```
   XAI_API_KEY=your_api_key_here
   ```
2. Go to **"Chat Mode"** tab
3. Describe your image in natural language
4. The AI will transform it into an optimized prompt and generate the image

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | No | xAI API key for Chat Mode prompt enhancement |

### Resolution Options

| Category | Available Ratios |
|----------|-----------------|
| 512px | 1:1, 5:3, 3:5, 4:3, 3:4, 16:9, 9:16 |
| 768px | 1:1, 5:3, 3:5, 4:3, 3:4, 16:9, 9:16 |
| 1024px | 1:1, 5:3, 3:5, 4:3, 3:4, 16:9, 9:16 |
| 1280px | 1:1, 3:2, 2:3, 16:9, 9:16 |

### Inference Steps

- **Default: 8** - Good balance of speed and quality
- **4-6** - Faster, slightly lower quality
- **10-20** - Higher quality, slower generation

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | Ventura 13.0+ | Sonoma 14.0+ |
| Chip | Apple M1 | Apple M2/M3/M4 |
| Memory | 16GB | 32GB+ |
| Storage | 10GB free | 20GB+ free |

## Technical Notes

This project includes several optimizations for Apple MPS:

- **bfloat16 precision** for transformer blocks
- **float32 VAE** to prevent NaN artifacts
- **CPU-based generator** for reproducible seeds
- **Attention slicing** to reduce memory usage
- **Memory cache clearing** after each generation

## Project Structure

```
z-image-inference/
├── main.py              # Gradio web interface
├── model_server.py      # FastAPI inference server
├── grok_client.py       # xAI Grok API client
├── i18n/                # Internationalization
│   ├── __init__.py
│   └── translations.yaml
├── pyproject.toml       # Project dependencies
└── history.db           # Generated images database
```

## Troubleshooting

**Model server won't start?**
- Ensure you have enough free memory (close other apps)
- First run downloads ~6GB model, be patient

**Black or corrupted images?**
- This is usually a MPS memory issue
- Try lowering resolution or closing other apps
- Restart the model server

**Chat mode not working?**
- Check if `XAI_API_KEY` is set in `.env`
- Verify your API key is valid

## Acknowledgments

- [Tongyi Lab](https://huggingface.co/Tongyi-MAI) for Z-Image-Turbo model
- [Hugging Face](https://huggingface.co/docs/diffusers) for diffusers library
- [Gradio](https://gradio.app/) for the web interface framework
- [z-image-mps](https://github.com/ivanfioravanti/z-image-mps) for MPS optimization inspiration

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

# Z-Image-Turbo Apple Silicon 本地运行

> 两条命令，在 Mac 上本地运行 Z-Image-Turbo。

## 功能特性

- **原生 Apple Silicon 支持** - 针对 MPS (Metal) 深度优化
- **快速推理** - 约 8 步即可生成高质量图像，bfloat16 加速
- **双输入模式**
  - Chat 模式：用自然语言描述，AI 自动优化 prompt
  - 直接输入模式：完全控制，直接输入详细 prompt
- **灵活分辨率** - 512px 到 1280px，多种宽高比
- **历史记录** - 所有生成的图片自动保存到本地数据库
- **中英双语界面**

## 快速开始

### 环境要求

- 搭载 Apple Silicon 芯片的 Mac（M1/M2/M3/M4）
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 包管理器

### 安装

```bash
# 克隆仓库
git clone https://github.com/OrdinarySF/z-image-inference.git
cd z-image-inference

# 安装依赖（uv 会自动处理一切）
uv sync
```

### 运行

**第一步：启动模型服务**（首次运行会下载约 6GB 模型）

```bash
uv run python model_server.py
```

**第二步：新开一个终端，启动 Web 界面**

```bash
uv run python main.py
```

浏览器打开 http://127.0.0.1:7860 即可使用！

## 使用方法

### 直接输入模式

适合想要完全控制 prompt 的用户：

1. 切换到 **"直接输入 Prompt"** 标签页
2. 输入详细的 prompt
3. 调整分辨率和步数
4. 点击 **"生成图片"**

### Chat 模式（可选）

需要 [xAI API 密钥](https://x.ai/) 来启用 AI prompt 增强：

1. 在项目根目录创建 `.env` 文件：
   ```
   XAI_API_KEY=your_api_key_here
   ```
2. 切换到 **"Chat 模式"** 标签页
3. 用自然语言描述你想要的图片
4. AI 会将其转化为优化的 prompt 并生成图片

## 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| macOS | Ventura 13.0+ | Sonoma 14.0+ |
| 芯片 | Apple M1 | Apple M2/M3/M4 |
| 内存 | 16GB | 32GB+ |
| 存储 | 10GB 可用空间 | 20GB+ 可用空间 |

## 致谢

- [通义实验室](https://huggingface.co/Tongyi-MAI) 的 Z-Image-Turbo 模型
- [Hugging Face](https://huggingface.co/docs/diffusers) 的 diffusers 库
- [Gradio](https://gradio.app/) Web 界面框架
- [z-image-mps](https://github.com/ivanfioravanti/z-image-mps) 提供 MPS 性能优化灵感

## 许可证

Apache License 2.0
