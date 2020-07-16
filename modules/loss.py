import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error


def discriminator_loss(real, generated):
    real_loss = mean_squared_error(tf.ones_like(real), real)
    generated_loss = mean_squared_error(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return mean_squared_error(tf.ones_like(generated), generated)


def cycle_loss(real_image, cycled_image, LAMBDA=10):
    loss = tf.math.reduce_mean(tf.math.abs(real_image - cycled_image))
    return loss * LAMBDA


def identity_loss(real_image, same_image, LAMBDA=10):
    loss = tf.math.reduce_mean(tf.math.abs(real_image - same_image))
    return loss * LAMBDA * 0.5
