# # Install required libraries
# !pip install tensorflow
# !pip install seaborn
# !pip install pandas

from tensorflow import keras
from tensorflow.keras.constraints import MaxNorm
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.utils import shuffle
import tensorflow as tf
import pickle
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_test.shape)

print(y_train[0])

plt.imshow(X_train[0])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# One hot encoding using keras.utils
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[0])

num_class = y_test.shape[1] #10
print(X_train.shape[1:])

# CNN Model
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=X_train.shape[1:], activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
model.add(MaxPool2D(2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
model.add(MaxPool2D(2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(num_class, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy'] )
model.summary()

# Define the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_dim=100))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Dense(np.prod(X_train.shape[1:]), activation='tanh'))
    model.add(tf.keras.layers.Reshape(X_train.shape[1:]))
    return model

# Define the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

# Build and compile the models
generator = build_generator()
discriminator = build_discriminator()

class WGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, discriminator_steps=5, gp_weight=10.0):
        super(WGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_steps = discriminator_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Computes the gradient penalty."""
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        # Train the discriminator
        for i in range(self.discriminator_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)

                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight

            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, 100))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            g_loss = self.g_loss_fn(fake_logits)

        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Define the loss functions
def d_loss_fn(real_img, fake_img):
    return tf.reduce_mean(fake_img) - tf.reduce_mean(real_img)

def g_loss_fn(fake_img):
    return -tf.reduce_mean(fake_img)

# Compile the WGAN model
wgan = WGAN(generator=generator, discriminator=discriminator)
wgan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    d_loss_fn=d_loss_fn,
    g_loss_fn=g_loss_fn,
)

# Training the WGAN
whistory=wgan.fit(X_train, batch_size=64, epochs=100)
for key,val in whistory.history.items():
  print(key)

pd.DataFrame(whistory.history).plot()
plt.plot(whistory.history['d_loss'])
plt.plot(whistory.history['g_loss'])
plt.show()

# Generate synthetic images
def generate_synthetic_images(generator, num_images):
    random_latent_vectors = tf.random.normal(shape=(num_images, 100))
    synthetic_images = generator(random_latent_vectors)
    return synthetic_images

# Generate synthetic images
num_synthetic_images = 5000
synthetic_images = generate_synthetic_images(generator, num_synthetic_images)

# Combine real and synthetic images
augmented_X_train = np.concatenate([X_train, synthetic_images])
augmented_y_train = np.concatenate([y_train, to_categorical(np.random.randint(0, num_class, num_synthetic_images), num_classes=num_class)])




# Train the CNN with the augmented dataset
history = model.fit(augmented_X_train, augmented_y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=[EarlyStopping(patience=100)])

for key,val in history.history.items():
  print(key)

pd.DataFrame(history.history).plot()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

model.save('CNN_cifar10_augmented.h5')
pickle.dump(model, open('./model_augmented.p', 'wb'))

# Continue from where we left off

model2 = load_model('CNN_cifar10_augmented.h5')

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Evaluate the model on test set
scores = model2.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Confusion matrix
y_pred = model2.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()







