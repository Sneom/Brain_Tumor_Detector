import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the trained model
model = load_model('brain.keras')

# Function to preprocess an image for prediction
def preprocess_image(image_path):
  """
  Loads an image, resizes it to the model's input shape,
  normalizes pixel values, and adds a batch dimension.
  """
  img = cv2.imread(image_path)
  if img is None:
      print(f"Error: Could not read image file: {image_path}")
      return None  # Indicate error

  img = Image.fromarray(img, 'RGB')
  img = img.resize((64, 64))  # Match your model's input shape

  img = np.array(img)
  img = img / 255.0  # Normalize pixel values

  img = np.expand_dims(img, axis=0)  # Add a batch dimension

  return img

# Get the path to your test image (modify as needed)
test_image_path = 'C:\\Users\\HP\\Downloads\\brain-main\\uploads\\pred9.jpg'

# Preprocess the test image
test_image = preprocess_image(test_image_path)

if test_image is not None:  # Check if preprocessing succeeded
  # Make a prediction
  prediction = model.predict(test_image)
  print(f"Prediction shape: {prediction.shape}")

  # Interpret the prediction based on its shape (assuming class 1 is "tumor")
  if prediction.shape == (1, 2):
    class_index = np.argmax(prediction[0])  # Get the index of the class with highest probability
    if class_index == 1:
      print("Prediction: Tumor")
    else:
      print("Prediction: No Tumor")
  else:
    print("Unexpected prediction shape. Modify code accordingly.")