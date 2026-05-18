"""Mobile-friendly web UI for Z-Image generation, accessible over local network."""
import base64
import random
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

DB_PATH = Path(__file__).parent / "history.db"
MODEL_SERVER_URL = "http://127.0.0.1:8000"
HTML_PATH = Path(__file__).parent / "mobile.html"
MANIFEST_PATH = Path(__file__).parent / "manifest.json"
SW_PATH = Path(__file__).parent / "sw.js"

RESOLUTION_OPTIONS = {
    "512": {
        "512x512 (1:1)": (512, 512),
        "640x384 (5:3)": (640, 384),
        "384x640 (3:5)": (384, 640),
        "512x384 (4:3)": (512, 384),
        "384x512 (3:4)": (384, 512),
        "640x368 (16:9)": (640, 368),
        "368x640 (9:16)": (368, 640),
    },
    "768": {
        "768x768 (1:1)": (768, 768),
        "960x576 (5:3)": (960, 576),
        "576x960 (3:5)": (576, 960),
        "768x576 (4:3)": (768, 576),
        "576x768 (3:4)": (576, 768),
        "960x544 (16:9)": (960, 544),
        "544x960 (9:16)": (544, 960),
    },
    "1024": {
        "1024x1024 (1:1)": (1024, 1024),
        "1280x768 (5:3)": (1280, 768),
        "768x1280 (3:5)": (768, 1280),
        "1024x768 (4:3)": (1024, 768),
        "768x1024 (3:4)": (768, 1024),
        "1280x720 (16:9)": (1280, 720),
        "720x1280 (9:16)": (720, 1280),
        "1024x576 (16:9)": (1024, 576),
        "576x1024 (9:16)": (576, 1024),
    },
    "1280": {
        "1280x1280 (1:1)": (1280, 1280),
        "1536x1024 (3:2)": (1536, 1024),
        "1024x1536 (2:3)": (1024, 1536),
        "1536x864 (16:9)": (1536, 864),
        "864x1536 (9:16)": (864, 1536),
    },
}

EXAMPLE_PROMPTS = [
    "一位男士和他的贵宾犬穿着配套的服装参加狗狗秀，室内灯光，背景中有观众。",
    "极具氛围感的暗调人像，一位优雅的中国美女在黑暗的房间里。一束强光通过遮光板，在她的脸上投射出一个清晰的闪电形状的光影，正好照亮一只眼睛。高对比度，明暗交界清晰，神秘感，莱卡相机色调。",
    "一张中景手机自拍照片拍摄了一位留着长黑发的年轻东亚女子在灯光明亮的电梯内对着镜子自拍。她穿着一件带有白色花朵图案的黑色露肩短上衣和深色牛仔裤。她的头微微倾斜，嘴唇嘟起做亲吻状，非常可爱俏皮。",
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights.",
]


# --- DB ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_data BLOB NOT NULL,
            prompt TEXT,
            seed INTEGER,
            width INTEGER,
            height INTEGER,
            steps INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(image_bytes, prompt, seed, width, height, steps):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO images (image_data, prompt, seed, width, height, steps) VALUES (?, ?, ?, ?, ?, ?)",
        (image_bytes, prompt, seed, width, height, steps),
    )
    conn.commit()
    conn.close()


# --- Task Queue ---

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.order = []
        self.pending = []
        self.lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, prompt, width, height, steps, seed, random_seed):
        tid = uuid.uuid4().hex[:8]
        task = {
            "id": tid,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "random_seed": random_seed,
            "status": "pending",
            "image_base64": None,
            "error": None,
            "created_at": time.time(),
            "completed_at": None,
        }
        with self.lock:
            self.tasks[tid] = task
            self.order.append(tid)
            self.pending.append(tid)
        return tid

    def cancel(self, tid):
        with self.lock:
            t = self.tasks.get(tid)
            if t and t["status"] == "pending":
                t["status"] = "cancelled"
                if tid in self.pending:
                    self.pending.remove(tid)
                return True
        return False

    def remove(self, tid):
        with self.lock:
            self.tasks.pop(tid, None)
            if tid in self.order:
                self.order.remove(tid)

    def get_all(self):
        with self.lock:
            return [dict(self.tasks[tid]) for tid in reversed(self.order) if tid in self.tasks]

    def get(self, tid):
        with self.lock:
            t = self.tasks.get(tid)
            return dict(t) if t else None

    def clear_done(self):
        with self.lock:
            done = [tid for tid, t in self.tasks.items()
                    if t["status"] in ("completed", "failed", "cancelled")]
            for tid in done:
                del self.tasks[tid]
                if tid in self.order:
                    self.order.remove(tid)

    def _run(self):
        while True:
            tid = None
            with self.lock:
                while self.pending:
                    candidate = self.pending.pop(0)
                    t = self.tasks.get(candidate)
                    if t and t["status"] == "pending":
                        t["status"] = "running"
                        tid = candidate
                        break

            if tid is None:
                time.sleep(0.3)
                continue

            task = self.tasks[tid]
            try:
                seed = task["seed"]
                if task["random_seed"]:
                    seed = random.randint(0, 2147483647)

                resp = requests.post(
                    f"{MODEL_SERVER_URL}/generate",
                    json={
                        "prompt": task["prompt"],
                        "width": task["width"],
                        "height": task["height"],
                        "steps": task["steps"],
                        "seed": seed,
                    },
                    timeout=3000,
                )
                resp.raise_for_status()
                data = resp.json()
                image_bytes = base64.b64decode(data["image_base64"])

                save_to_db(image_bytes, task["prompt"], seed,
                           task["width"], task["height"], task["steps"])

                with self.lock:
                    task["status"] = "completed"
                    task["image_base64"] = data["image_base64"]
                    task["seed"] = seed
                    task["completed_at"] = time.time()

            except Exception as e:
                with self.lock:
                    task["status"] = "failed"
                    task["error"] = str(e)
                    task["completed_at"] = time.time()


task_manager = TaskManager()


# --- FastAPI ---

app = FastAPI()


class SubmitRequest(BaseModel):
    prompt: str
    res_category: str = "512"
    resolution: str = "368x640 (9:16)"
    seed: int = -1
    random_seed: bool = True
    steps: int = 8
    count: int = 1


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/manifest.json")
def manifest():
    return Response(content=MANIFEST_PATH.read_bytes(), media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return Response(content=SW_PATH.read_bytes(), media_type="application/javascript")


@app.get("/api/icon/{size}")
def icon(size: int):
    # Minimal 1x1 blue PNG scaled by the browser for the PWA icon
    import struct, zlib
    s = min(size, 512)
    # Generate a simple blue square PNG
    raw = b''
    for _ in range(s):
        raw += b'\x00' + b'\x00\x7a\xff' * s
    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', s, s, 8, 2, 0, 0, 0)
    img = (b'\x89PNG\r\n\x1a\n'
           + png_chunk(b'IHDR', ihdr)
           + png_chunk(b'IDAT', zlib.compress(raw))
           + png_chunk(b'IEND', b''))
    return Response(content=img, media_type="image/png")


@app.get("/api/config")
def config():
    return {
        "resolutions": {cat: list(opts.keys()) for cat, opts in RESOLUTION_OPTIONS.items()},
        "examples": [{"title": p[:18] + "..." if len(p) > 18 else p, "prompt": p} for p in EXAMPLE_PROMPTS],
    }


@app.post("/api/submit")
def submit(req: SubmitRequest):
    w, h = RESOLUTION_OPTIONS.get(req.res_category, {}).get(
        req.resolution, (1024, 1024))
    ids = []
    for _ in range(min(req.count, 20)):
        ids.append(task_manager.submit(
            req.prompt, w, h, req.steps, req.seed, req.random_seed))
    return {"task_ids": ids}


@app.get("/api/tasks")
def list_tasks():
    tasks = task_manager.get_all()
    out = []
    for t in tasks:
        r = {k: v for k, v in t.items() if k != "image_base64"}
        r["has_image"] = t["image_base64"] is not None
        out.append(r)
    return out


@app.get("/api/task/{tid}/image")
def task_image(tid: str):
    t = task_manager.get(tid)
    if not t or not t["image_base64"]:
        raise HTTPException(404)
    return Response(content=base64.b64decode(t["image_base64"]), media_type="image/png")


@app.delete("/api/task/{tid}")
def cancel_task(tid: str):
    if not task_manager.cancel(tid):
        task_manager.remove(tid)
    return {"ok": True}


@app.delete("/api/tasks/done")
def clear_done():
    task_manager.clear_done()
    return {"ok": True}


@app.get("/api/history")
def list_history(limit: int = 50, offset: int = 0):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, prompt, seed, width, height, steps, created_at "
        "FROM images ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    conn.close()
    return {
        "items": [
            {"id": r[0], "prompt": r[1], "seed": r[2], "width": r[3],
             "height": r[4], "steps": r[5], "created_at": r[6]}
            for r in rows
        ],
        "total": total,
    }


@app.get("/api/history/{img_id}/image")
def history_image(img_id: int):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT image_data FROM images WHERE id = ?", (img_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404)
    return Response(content=row[0], media_type="image/png")


@app.delete("/api/history/{img_id}")
def delete_history(img_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM images WHERE id = ?", (img_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/history")
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/health")
def health():
    try:
        r = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "error", "model_loaded": False}


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


if __name__ == "__main__":
    init_db()
    ip = get_local_ip()
    print(f"\n  Mobile UI ready:")
    print(f"  Local:   http://localhost:8080")
    print(f"  iPhone:  http://{ip}:8080\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)
