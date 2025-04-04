import numpy as np
import tensorflow as tf
from keras.applications import VGG19
from keras.models import Model
from keras.utils import load_img, img_to_array
from feature_extraction import load_style_features
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img_array)

def deprocess_image(x):
    x = x.reshape((IMG_HEIGHT, IMG_WIDTH, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

def neural_style_transfer(content_path, style_name, num_iterations=100, content_weight=1e3, style_weight=1e-2):
    # Load content and style features
    content_image = preprocess_image(content_path)
    style_features = load_style_features(style_name)

    # Load VGG19
    model = VGG19(include_top=False, weights='imagenet')
    outputs = [model.get_layer(name).output for name in ['block5_conv2', 'block5_conv4']]
    feature_extractor = Model(inputs=model.input, outputs=outputs)

    content_target, _ = feature_extractor.predict(content_image)
    style_target = gram_matrix(style_features)

    generated_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            content_output, style_output = feature_extractor(generated_image)
            content_loss = tf.reduce_mean((content_output - content_target) ** 2)
            gram_generated = gram_matrix(style_output)
            style_loss = tf.reduce_mean((gram_generated - style_target) ** 2)
            total_loss = content_weight * content_loss + style_weight * style_loss

        grad = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -128.0, 128.0))

    for i in range(num_iterations):
        train_step()

    final_img = deprocess_image(generated_image.numpy())
    return Image.fromarray(final_img)

def apply_prompt_style(image_path, prompt):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    original_image = Image.open(image_path).convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=original_image).images[0]
    return result

def apply_style(content_image_path, style_or_prompt, use_prompt=False):
    if use_prompt:
        return apply_prompt_style(content_image_path, style_or_prompt)
    else:
        return neural_style_transfer(content_image_path, style_or_prompt)
