import numpy as np
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_generator(latent_dim, output_dim):
    inputs = Input(shape=(latent_dim,))
    x = Dense(128)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(output_dim, activation='tanh')(x)
    return Model(inputs, outputs)

def build_discriminator(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(512)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

def compile_gan(generator, discriminator, latent_dim):
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    fake_data = generator(gan_input)
    gan_output = discriminator(fake_data)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
    return gan

def train_gan(generator, discriminator, gan, X_train_scaled, latent_dim, batch_size=128, epochs=10000):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
        real_data = X_train_scaled[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

if __name__ == "__main__":
    latent_dim = 100
    X_train_scaled = np.load('X_train_scaled.npy')  # Load preprocessed data
    generator = build_generator(latent_dim, X_train_scaled.shape[1])
    discriminator = build_discriminator(X_train_scaled.shape[1])
    gan = compile_gan(generator, discriminator, latent_dim)
    train_gan(generator, discriminator, gan, X_train_scaled, latent_dim)
