import tensorflow as tf
import numpy as np
import PIL.Image
from image_preprocessing import load_img, tensor_to_image

# Load Pre-trained VGG19 Model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

# Function to Extract Style and Content Features
def get_feature_model():
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    return model

feature_extractor = get_feature_model()

def extract_features(image_path):
    image = load_img(image_path)
    image = tf.keras.applications.vgg19.preprocess_input(image * 255)
    features = feature_extractor(image)
    style_features = features[:num_style_layers]
    content_features = features[num_style_layers:]
    return style_features, content_features

# Dictionary to Store Extracted Style Features
style_feature_dict = {
    'Van Gough - Starry Night': extract_features('assets/styles/starry_night.jpg')[0],
    'Da Vinci - Mona Lisa': extract_features('assets/styles/mona_lisa.jpg')[0],
    'Picasso - Cubism': extract_features('assets/styles/cubism.jpg')[0],
    'Claude Monet - Impressionism': extract_features('assets/styles/impressionism.jpg')[0],
    'Salvador Dali - Surrealism': extract_features('assets/styles/surrealism.jpg')[0],
    'Cyberpunk': extract_features('assets/styles/cyberpunk.jpg')[0]
}

# Function to Compute Gram Matrix (Style Representation)
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Function to Apply Style Transfer
def apply_style(content_path, style_name):
    content_image = load_img(content_path)
    content_features = extract_features(content_path)[1]
    style_features = style_feature_dict[style_name]
    
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.02)
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            generated_features = feature_extractor(generated_image)
            generated_style_features = generated_features[:num_style_layers]
            generated_content_features = generated_features[num_style_layers:]
            
            style_loss = tf.add_n([tf.reduce_mean(tf.square(gram_matrix(gen) - gram_matrix(orig)))
                                   for gen, orig in zip(generated_style_features, style_features)])
            
            content_loss = tf.reduce_mean(tf.square(generated_content_features[0] - content_features[0]))
            
            total_loss = style_loss * 1e-2 + content_loss * 1e4
        
        grads = tape.gradient(total_loss, generated_image)
        opt.apply_gradients([(grads, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
    
    for _ in range(100):
        train_step()
    
    return tensor_to_image(generated_image)
