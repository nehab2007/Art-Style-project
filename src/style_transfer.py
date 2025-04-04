import os
import torch
import numpy as np
from PIL import Image
from keras.applications import VGG19
from keras.models import Model
from keras.utils import img_to_array, load_img
from keras.applications.vgg19 import preprocess_input

from diffusers import StableDiffusionImg2ImgPipeline
from feature_extraction import load_style_features
from diffusers import StableDiffusionPipeline

# --- Load Stable Diffusion model with Hugging Face token ---
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=True
).to("cuda" if torch.cuda.is_available() else "cpu")


# --- VGG19-based Neural Style Transfer (for predefined styles) ---
def neural_style_transfer(content_image_path, style_name):
    content_img = load_img(content_image_path, target_size=(224, 224))
    content_array = img_to_array(content_img)
    content_array = np.expand_dims(content_array, axis=0)
    content_array = preprocess_input(content_array)

    # Load style features
    style_features = load_style_features(style_name)  # shape: (1, 14, 14, 512)

    # Just blend features (simple version)
    # This is not full style transfer but a basic filter blending
    blended_array = 0.5 * content_array  # Dummy combination

    # Decode array back to image
    blended_array = np.clip(blended_array[0], 0, 255).astype('uint8')
    blended_image = Image.fromarray(blended_array)

    return blended_image

# --- Stable Diffusion-based Transfer (for custom prompts) ---
def prompt_based_style_transfer(uploaded_image_path, prompt):
    init_image = Image.open(uploaded_image_path).convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=init_image, strength=0.7, guidance_scale=7.5)
    styled_image = result.images[0]
    return styled_image

# --- Master function to handle both methods ---
def apply_style(content_image_path, style_or_prompt, is_prompt=False):
    if is_prompt:
        return prompt_based_style_transfer(content_image_path, style_or_prompt)
    else:
        return neural_style_transfer(content_image_path, style_or_prompt)
