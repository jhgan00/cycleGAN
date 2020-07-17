from tensorflow.keras.optimizers import Adam

generator_g_optimizer = Adam(2e-4, beta_1=0.5)
generator_f_optimizer = Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = Adam(2e-5, beta_1=0.5)
discriminator_y_optimizer = Adam(2e-5, beta_1=0.5)