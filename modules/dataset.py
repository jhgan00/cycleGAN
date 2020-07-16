import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BUFFER_SIZE = 128
BATCH_SIZE = 32
WIDTH = 256
HEIGHT = 256

def random_crop(image):
    return tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])

def normalize(image):
    return (tf.cast(image, tf.float32) / 127.5 ) - 1

def random_jitter(image):
    image = tf.image.resize(image, [286, 286], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image):
    image = normalize(image)
    return image

def build_generator(path, batch_size=32, class_mode=None):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image_train,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.2,
        vertical_flip=True
    )
    generator = datagen.flow_from_directory(
        path,
        class_mode=class_mode,
        batch_size=batch_size
    )
    return generator

real_dataset = tf.data.Dataset.from_generator(
    lambda: build_generator("./sample/real-world"),
    output_types=tf.float32,
    output_shapes=[32, 256, 256, 3]
)

tattoo_dataset = tf.data.Dataset.from_generator(
    lambda: build_generator("./sample/tattoo"),
    output_types=tf.float32,
    output_shapes=[32, 256, 256, 3]
)