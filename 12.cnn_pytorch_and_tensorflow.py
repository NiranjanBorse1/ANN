# CNN using MNIST Dataset in TensorFlow

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images for CNN
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Create CNN Model
model = models.Sequential()

# Convolution Layer
model.add(layers.Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=(28,28,1)))

# Max Pooling Layer
model.add(layers.MaxPooling2D((2,2)))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(64, activation='relu'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_images, train_labels, epochs=5)

# Evaluate Model
loss, accuracy = model.evaluate(test_images, test_labels)

print("Test Accuracy:", accuracy)