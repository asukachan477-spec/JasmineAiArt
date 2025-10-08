from realesrgan import RealESRGAN
from PIL import Image

def upscale_image(image: Image.Image) -> Image.Image:
    model = RealESRGAN(device="cuda")
    model.load_weights("RealESRGAN_x4.pth")
    return model.predict(image)
