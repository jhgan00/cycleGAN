from tensorflow.keras.optimizers import Adam

gen_G_optimizer = Adam(2e-4, beta_1=0.5)
gen_F_optimizer = Adam(2e-4, beta_1=0.5)

disc_X_optimizer = Adam(2e-4, beta_1=0.5)
disc_Y_optimizer = Adam(2e-4, beta_1=0.5)