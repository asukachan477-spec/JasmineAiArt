from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.lora import get_loras
from utils.upscale import upscale_image
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pipeline initialisieren
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16
).to("cuda")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    loras = get_loras()
    return templates.TemplateResponse("index.html", {"request": request, "loras": loras})

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    lora1: str = Form(None),
    lora2: str = Form(None),
    steps: int = Form(20),
    upscale: bool = Form(False)
):
    # TODO: Lora Mixing implementieren
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]

    if upscale:
        image = upscale_image(image)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    loras = get_loras()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image": img_str,
        "prompt": prompt,
        "loras": loras
