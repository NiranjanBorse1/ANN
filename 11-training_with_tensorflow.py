# Logistic Regression using TensorFlow

import tensorflow as tf
import numpy as np

# Input data
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

# Output data
Y = np.array([
    [0],
    [1],
    [1],
    [1]
], dtype=float)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, Y, epochs=1000, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(X, Y, verbose=0)

print("Accuracy:", accuracy)

# Predictions
predictions = model.predict(X)

print("Predictions:")
print(np.round(predictions))