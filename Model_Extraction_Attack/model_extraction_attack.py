import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras import metrics, losses
from keras.models import load_model
import urllib.request
import os
import tarfile
from PIL import Image
import glob

# IMPORTS
print("Loading required libraries...")

def query_teacher_model(model, input_data):
    """
    Simulates black-box access to the teacher model.
    Only returns predictions, not the model itself.
    """
    return model.predict(input_data)

# Load the teacher model (simulating a black-box service)
print("Loading teacher model...")
try:
    teacher_model = load_model('teacher_model.h5')
    teacher_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    print("Teacher model loaded successfully!")
except Exception as e:
    print(f"Error loading teacher model: {str(e)}")

# Create a new student model (this would be our attack model)
print("Creating student model...")
student_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Load STL-10 dataset for the attack
print("Loading STL-10 dataset...")
train_images = []
train_path = 'STL-10/train_images'
for img_path in glob.glob(os.path.join(train_path, '*.*')):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    train_images.append(img_array)
X_train = np.array(train_images)

test_images = []
test_path = 'STL-10/test_images'
for img_path in glob.glob(os.path.join(test_path, '*.*')):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    test_images.append(img_array)
X_test = np.array(test_images)

# Normalize the data
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# Create labels (0-9 for 10 classes)
y_train = np.zeros(len(X_train))
y_test = np.zeros(len(X_test))

# Convert labels to one-hot encoding
test_labels = to_categorical(y_test)
train_labels = to_categorical(y_train)

# Simulate the attack
print("Starting model extraction attack...")

# 1. Query the teacher model (black-box access)
print("Querying teacher model for predictions...")
teacher_predictions = query_teacher_model(teacher_model, X_train)

# 2. Train student model using only these predictions
print("Training student model using teacher predictions...")
student_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train the student model using the predictions
print("Training extracted model...")
history = student_model.fit(
    X_train,
    teacher_predictions,  # Use teacher's predictions instead of true labels
    epochs=7,
    batch_size=32,
    validation_split=0.2
)

# 4. Evaluate the extracted model
print("Evaluating extracted model...")
extracted_model_metrics = student_model.evaluate(X_test, test_labels)
print(f"Extracted model metrics: {extracted_model_metrics}")

# Save the extracted model
print("Saving extracted model...")
student_model.save('extracted_model.h5')
print("Attack completed successfully!") 