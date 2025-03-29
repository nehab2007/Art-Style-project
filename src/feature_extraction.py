import tensorflow as tf
import numpy as np
import os
from image_processing import load_img

# Load the pre-trained VGG19 model without the classification layers
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

def extract_features(image_path):
    """
    Extracts features from an image using VGG19.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Extracted feature vector.
    """
    image = load_img(image_path, max_dim=224)  # Resize to 224x224 for VGG19
    features = base_model.predict(image)
    return np.squeeze(features)  # Remove batch dimension

# Define path for predefined images
style_images_path = 'Art style transfer project/assets/styles/'
output_features_path = 'Art style transfer project/assets/features/'

# Ensure output directory exists
os.makedirs(output_features_path, exist_ok=True)

# List all style images
style_images = {
    'Van_Gogh': 'Starry_Night.jpg',
    'Da_Vinci': 'Mona_Lisa.jpg',
    'Picasso': 'Cubism.jpg',
    'Monet': 'Impressionism.jpg',
    'Dali': 'Surrealism.jpg',
    'Cyberpunk': 'Cyberpunk.jpg'
}

# Extract and save features
for style, filename in style_images.items():
    img_path = os.path.join(style_images_path, filename)
    if os.path.exists(img_path):
        features = extract_features(img_path)
        np.save(os.path.join(output_features_path, f"{style}_features.npy"), features)
        print(f"Extracted and saved features for {style}")
    else:
        print(f"Image not found: {img_path}")