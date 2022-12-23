auth_token = "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"
#please obtain auth_token from huggingface first and replace.

"""
please install required dependencies first
pip install fastapi
pip install torch
pip install --upgrade diffusers[torch]
pip install transformers
"""
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

def dummy(images, **kwargs):
    return images, False

app = FastAPI()
device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

pipe.safety_checker = dummy

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.get("/")
def generate(prompt: str): 
    with autocast(device): 
        image = pipe(prompt, guidance_scale=8.5, height=256, width=256).images[0] #change height and width to 512 for use with powerful gpu

    image.save("testimage.png")
    #image generated with nvidia 16XX gpu will only be a green image.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
    