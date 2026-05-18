import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import io
import base64
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import ZImagePipeline


pipe = None


def load_model():
    global pipe
    if pipe is None:
        print("Loading model...")
        # 1. 基础加载
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            low_cpu_mem_usage=True
        )
        
        # 2. 针对 Mac M4 优化精度和设备
        # 使用 bfloat16 能极大缓解黑屏问题
        print("Moving model to MPS with torch.bfloat16...")
        pipe.to(device="mps", dtype=torch.bfloat16) 
        
        # 3. 优化 VAE：这是防止黑屏的关键。
        # 虽然去掉了 tiling，但 VAE 强制转 float32 必须保留
        pipe.vae.to(device="mps", dtype=torch.float32)
        
        # 4. 禁用安全检查（解决你之前的 NSFW 报错）
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
        
        # 注意：这里删掉了报错的 enable_vae_tiling()
        # 如果你担心显存，可以尝试保留下面这一行（通常 ZImagePipeline 支持）
        try:
            pipe.enable_attention_slicing()
        except:
            pass

        print("Model loaded on M4 GPU (MPS)!")

    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 8
    seed: int = -1


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = load_model()

    # 将 seed 处理好
    seed = req.seed if req.seed != -1 else torch.seed()
    generator = torch.Generator("cpu").manual_seed(seed)

    # 【重要】默认分辨率先设小一点测试，比如 512
    width = req.width if req.width > 0 else 512
    height = req.height if req.height > 0 else 512

    with torch.inference_mode():
        # 【核心改动 4】确保 guidance_scale 为 0.0 是该模型的要求
        image = model(
            prompt=req.prompt,
            height=height,
            width=width,
            num_inference_steps=req.steps,
            guidance_scale=0.0,
            max_sequence_length=512,
            generator=generator,
        ).images[0]

    # 转换为 base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    torch.mps.empty_cache()

    return GenerateResponse(image_base64=image_base64, seed=req.seed)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)