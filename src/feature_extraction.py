import os
import pickle
import numpy as np
import tensorflow as tf
from keras.applications import VGG19
from keras.utils import load_img, img_to_array
from keras.models import Model

def extract_style_features(style_image_path, style_name, output_dir="assets/styles_features"):
    """Extract and save style features from a given style image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = VGG19(weights='imagenet', include_top=False)
    feature_extractor = Model(
        inputs=model.input,
        outputs=[model.get_layer("block5_conv4").output]
    )

    # Load and preprocess style image
    img = load_img(style_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

    # Extract style features
    extracted_features = feature_extractor.predict(img_array)

    # Save extracted features to a pickle file
    feature_file = os.path.join(output_dir, f"{style_name.replace(' ', '_')}.pkl")
    with open(feature_file, 'wb') as f:
        pickle.dump(extracted_features, f)

    print(f"[✔] Extracted features saved: {feature_file}")
    return feature_file

def load_style_features(style_name, output_dir="assets/styles_features"):
    """Load style features from the pickle file."""
    feature_file = os.path.join(output_dir, f"{style_name.replace(' ', '_')}.pkl")
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            extracted_features = pickle.load(f)
        return extracted_features
    else:
        raise ValueError(f"[✘] No extracted features found for {style_name}.")

# Automatically extract all predefined styles
styles = {
    "Van Gogh - Starry Night": "assets/styles/Van Gogh - Starry Night.jpg",
    "Salvador Dali - Surrealism": "assets/styles/Salvador Dali - Surrealism.jpg",
    "Picasso - Cubism": "assets/styles/Picasso - Cubism.jpg",
    "Da Vinci - Mona Lisa": "assets/styles/Da Vinci - Mona Lisa.jpg",
    "Claude Monet - Impressionism": "assets/styles/Claude Monet - Impressionism.jpg",
    "Cyberpunk": "assets/styles/Cyberpunk.jpg"
}

if __name__ == "__main__":
    for style_name, image_path in styles.items():
        extract_style_features(image_path, style_name)
