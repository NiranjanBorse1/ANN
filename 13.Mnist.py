#MNIST Handwritten Digit Detection using TensorFlow/Keras
# MNIST Handwritten Digit Detection

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Create CNN model
model = models.Sequential()

# Convolution layer
model.add(layers.Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=(28,28,1)))

# Pooling layer
model.add(layers.MaxPooling2D((2,2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layer
model.add(layers.Dense(64, activation='relu'))

# Output layer
model.add(layers.Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=5)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)

print("Test Accuracy:", accuracy)




#MNIST Handwritten Digit Detection using PyTorch
# MNIST Digit Detection using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transform)

test_data = datasets.MNIST(root='./data',
                           train=False,
                           download=True,
                           transform=transform)

train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# CNN Model
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*13*13, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32*13*13)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

print("Training Completed")