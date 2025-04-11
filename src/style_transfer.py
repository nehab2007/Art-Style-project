import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np
import os

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-1.5"

    print("[INFO] Loading Stable Diffusion Img2Img model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=True)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # Disable safety checker just in case
    return pipe

PREDEFINED_STYLES = {
    "Van Gogh - Starry Night": "inspired by Van Gogh's Starry Night, with swirling brushstrokes and vibrant blues and yellows.",
    "Da Vinci - Mona Lisa": "in the style of Da Vinci’s Mona Lisa, with soft shading and Renaissance portrait techniques.",
    "Picasso - Cubism": "inspired by Picasso’s Cubism, with geometric abstraction and bold color contrasts.",
    "Claude Monet - Impressionism": "in Monet’s Impressionist style, with soft colors and delicate brush strokes.",
    "Salvador Dali - Surrealism": "in the surrealist style of Salvador Dali, featuring dreamlike, distorted elements.",
    "Cyberpunk": "in a futuristic cyberpunk aesthetic, with neon lights and dark dystopian themes."
}

def preprocess_image(image_path, target_size=(512, 512)):
    print(f"[INFO] Preprocessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image

def apply_style(input_image_path, selected_style, user_prompt):
    pipe = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess image
    init_image = preprocess_image(input_image_path)

    # Compose prompt
    style_description = PREDEFINED_STYLES.get(selected_style, f"in {selected_style} style")
    full_prompt = f"{style_description} {user_prompt}" if user_prompt else style_description
    print(f"[INFO] Final Prompt: {full_prompt}")

    # Generate styled image
    try:
        output = pipe(
            prompt=full_prompt,
            image=init_image,
            strength=0.55,  # Try 0.5–0.8 range
            guidance_scale=6.0,
            num_inference_steps=50
        ).images[0]
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return None

    # Save styled output
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "styled_image.png")
    output.save(output_path)
    print(f"[SUCCESS] Styled image saved at: {output_path}")

    # === Evolution Matrix (Only 5x5 for brevity) ===
    before_matrix = np.array(init_image)
    after_matrix = np.array(output)

    print("\n[EVOLUTION MATRIX SAMPLE] (first 5x5 pixels only):")
    print("\n[BEFORE]")
    print(before_matrix[:5, :5])

    print("\n[AFTER]")
    print(after_matrix[:5, :5])

    return output_path

if __name__ == "__main__":
    test_image = "test_input.jpg"  # Replace with an actual file to test
    test_style = "Cyberpunk"
    test_prompt = "Add glowing neon outlines"
    apply_style(test_image, test_style, test_prompt)
