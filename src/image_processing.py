import tensorflow as tf
import numpy as np
import PIL.Image

def load_img(img_path, max_dim=512):
    """
    Loads and preprocesses an image from a given path.
    - Rescales the image while maintaining aspect ratio.
    - Converts it to a float32 tensor.

    Args:
        img_path (str): Path to the image.
        max_dim (int): Maximum dimension to resize to.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize while maintaining aspect ratio
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    return tf.expand_dims(img, axis=0)

def tensor_to_image(tensor):
    """
    Converts a TensorFlow tensor to a PIL Image.

    Args:
        tensor (tf.Tensor): Tensor representing an image.

    Returns:
        PIL.Image: Converted image.
    """
    tensor = np.array(tensor * 255, dtype=np.uint8)
    if tensor.ndim > 3:
        tensor = tensor[0]  # Remove batch dimension
    return PIL.Image.fromarray(tensor)
