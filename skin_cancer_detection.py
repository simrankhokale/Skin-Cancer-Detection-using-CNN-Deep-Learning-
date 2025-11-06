# Skin Cancer Detection (Demo Version - No Dataset Needed)
# Author: Simran Khokale

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Create dummy dataset (100 images of size 64x64 with 3 color channels)
X_train = np.random.rand(100, 64, 64, 3)
y_train = np.random.randint(0, 2, 100)

X_test = np.random.rand(20, 64, 64, 3)
y_test = np.random.randint(0, 2, 20)

# Normalize images
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model (on random data)
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# Plot training graph
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training Progress (Demo)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict on one random image
sample = np.random.rand(1, 64, 64, 3)
prediction = model.predict(sample)
if prediction > 0.5:
    print("🔬 Prediction: Cancer Detected (Malignant)")
else:
    print("🌿 Prediction: Healthy Skin (Benign)")
