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
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        # VAE 用 float32 避免 bfloat16 导致的 NaN 问题
        pipe.vae.to(dtype=torch.float32)
        pipe.vae.config.force_upcast = True

        pipe.to(torch.device("mps"))
        pipe.enable_attention_slicing()
        print("Model loaded!")
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

    # MPS 的 Generator 放在 CPU 上更稳定
    generator = torch.Generator("cpu").manual_seed(req.seed)

    with torch.inference_mode():
        image = model(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
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
