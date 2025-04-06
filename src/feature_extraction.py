from transformers import CLIPFeatureExtractor
from PIL import Image
import torch
import os

# Set the path where your style images are stored
STYLE_FOLDER = "assets/styles/"
OUTPUT_FOLDER = "assets/features/"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the CLIP feature extractor
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

# Loop through each image in the style folder
for filename in os.listdir(STYLE_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        style_path = os.path.join(STYLE_FOLDER, filename)

        # Load and preprocess image
        image = Image.open(style_path).convert("RGB").resize((512, 512))
        features = feature_extractor(images=image, return_tensors="pt")["pixel_values"]

        # Save tensor
        output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.pt")
        torch.save(features, output_path)

        print(f"[✓] Extracted and saved features for: {filename}")
 
