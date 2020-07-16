import tensorflow as tf
from modules.loss import cycle_loss, discriminator_loss, generator_loss, identity_loss
from modules.model import generator_f, generator_g, discriminator_x, discriminator_y
from modules.optimizer import generator_f_optimizer, generator_g_optimizer, discriminator_x_optimizer, discriminator_y_optimizer
from modules.dataset import real_dataset, tattoo_dataset
from modules.checkpoint import ckpt_manager, ckpt
from tqdm import tqdm
import time


@tf.function
def train_step(real_x, real_y):

    with tf.GradientTape(persistent=True) as tape:
        # Generate Fake & Cycled Images
        fake_y = generator_g(real_x, training=True) # G(x)
        cycled_y = generator_f(fake_y, training=True) # F(G(x))

        fake_x = generator_f(real_y, training=True) # F(y)
        cycled_x = generator_g(fake_x, training=True) # G(F(y))

        # Generate Identity Images
        same_x = generator_f(real_x, training=True) # F(x)
        same_y = generator_g(real_y, training=True) # G(y)

        # Discriminate Real Images
        disc_real_x = discriminator_x(real_x, training=True) # D_x(x)
        disc_real_y = discriminator_y(real_y, training=True) # D_y(y)

        # Discriminate Fake Images
        disc_fake_x = discriminator_x(fake_x, training=True) # D_x(F(y))
        disc_fake_y = discriminator_y(fake_y, training=True) # D_y(G(x))

        # Calculate Generator Loss
        gen_g_loss = generator_loss(disc_fake_x) # G loss
        gen_f_loss = generator_loss(disc_fake_y) # F loss

        total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y) # Cycle Consistency Loss

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        # Calculate Discriminator loss
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Get Gradients for Each Networks
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizers
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, fake_x, fake_y


def train(max_epochs=100, tensorboard_path="logs/train"):
    print("Start Training ...")

    print("Searching Checkpoint Directory ...")

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        current_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1]) + 1
        print('[*] Latest checkpoint restored')

    else:
        print("[!] No checkpoint found")
        current_epoch = 0

    for epoch in range(current_epoch, max_epochs):
        start = time.time()

        with tqdm(total=8) as progress_bar:
            for image_x, image_y in tf.data.Dataset.zip((real_dataset, tattoo_dataset)):
                G_loss, F_loss, Dx_loss, Dy_loss, fake_x, fake_y = train_step(image_x, image_y)
                progress_bar.update()

        train_summary_writer = tf.summary.create_file_writer(tensorboard_path)

        with train_summary_writer.as_default():
            tf.summary.scalar("G_loss", gen_g_loss)
            tf.summary.scalar("F_loss", gen_f_loss)
            tf.summary.scalar("Dx_loss", disc_x_loss)
            tf.summary.scalar("Dy_loss", disc_y_loss)
            tf.summary.image("Fake Real Image", fake_x)
            tf.summary_image("Fake Tattoo Image", fake_y)

        print(
            f"EPOCH: {epoch + 1}  L(G): {gen_g_loss.numpy().round(3)}  L(F): {gen_f_loss.numpy().round(3)}  L(Dx): {disc_x_loss.numpy().round(3)}  L(Dy): {gen_y_loss.numpy().round(3)}")

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))

if __name__=="__main__":
    train()