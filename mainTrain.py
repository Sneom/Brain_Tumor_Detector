import cv2
import os
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image directory paths (modify as needed)
base_dir = 'datasets'
train_dir = os.path.join(base_dir, 'train')  # Assuming "train" folder has "yes" and "no" subfolders
test_dir = os.path.join(base_dir, 'test')   # Assuming "test" folder has "yes" and "no" subfolders

# Data augmentation for improved generalization (optional)
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define generators for training and validation data (categorical classification)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Two classes (yes/no)
)

validation_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Maintain order for evaluation
)

# Suppress potential UnicodeEncodeError (optional)
import sys
sys.stdout.reconfigure(encoding='utf-8')  # Set output encoding to UTF-8

# Define the CNN model architecture (using recommended input shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(2, activation='softmax'))  # Output layer with softmax for categorical classification

# Compile the model (consider class weights if your data is imbalanced)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (ensure all epochs run)
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)

# Save the trained model
model.save('brain.keras')