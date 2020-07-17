import tensorflow as tf
from tensorflow.data import Dataset

BATCH_SIZE = 8
WIDTH = 256
HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.resize(img, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def random_crop(image):
    return tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])


def random_jitter(image):
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def normalize(image):
    return (tf.cast(image, tf.float32) / 127.5) - 1


real_dataset = Dataset.list_files("data/real-world/*") \
    .map(process_path, num_parallel_calls=AUTOTUNE) \
    .map(random_jitter, num_parallel_calls=AUTOTUNE) \
    .map(normalize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

tattoo_dataset = Dataset.list_files("data/tattoo/*") \
    .map(process_path, num_parallel_calls=AUTOTUNE) \
    .map(random_jitter, num_parallel_calls=AUTOTUNE) \
    .map(normalize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
