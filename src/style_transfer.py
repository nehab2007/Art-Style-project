import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms

# Optional: Style prompt presets
PREDEFINED_STYLES = {
    "Van Gogh - Starry Night": "a painting in the style of Van Gogh's Starry Night",
    "Da Vinci - Mona Lisa": "a portrait inspired by Da Vinci's Mona Lisa",
    "Picasso - Cubism": "an abstract cubist artwork like Picasso",
    "Claude Monet - Impressionism": "an impressionist painting like Claude Monet",
    "Salvador Dali - Surrealism": "a surrealist piece like Salvador Dali",
    "Cyberpunk": "a futuristic cyberpunk-style digital painting"
}

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor
import torch

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor
import torch

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Jiali/stable-diffusion-1.5"

    print("Loading Stable Diffusion model without safety checker...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=None,  # disables NSFW filtering
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to(device)
    return pipe

def apply_style(input_image_path, selected_style, user_prompt):
    pipe = load_model()
    
    # Load and resize image
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    # Build prompt
    style_prompt = PREDEFINED_STYLES.get(selected_style, "")
    full_prompt = f"{style_prompt}. {user_prompt}".strip()
    print(f"Final Prompt: {full_prompt}")
    
    # Generate output
    output = pipe(prompt=full_prompt, image=init_image, strength=0.8, guidance_scale=7.5).images[0]
    
    # Save output
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "styled_output.png")
    output.save(output_path)
    print(f"Styled image saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    test_image = "input_image_path"  # Replace with dynamic input path
    test_style = "Claude Monet - Impressionism"
    test_prompt = "add cherry blossom trees and soft morning light"
    apply_style(test_image, test_style, test_prompt)
