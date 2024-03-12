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

    # Handle prediction based on its shape
    if prediction.shape == (1, 2):
        class_probability = prediction[0][1]  # Assuming class 1 is "tumor" for 2D output
    elif prediction.shape == (1, 1):
        class_probability = prediction[0][0]  # Access single probability value
    else:
        print("Unexpected prediction shape. Modify code accordingly.")
  # Interpret the prediction (assuming class probabilities):
  #   prediction[0][0] -> Probability of class 0 (likely no tumor)
  #   prediction[0][1] -> Probability of class 1 (likely tumor)
class_probability = prediction[0][1]  # Assuming class 1 is "tumor"
tumor_likelihood = round(class_probability * 100, 2)  # Percentage

print(f"Brain tumor likelihood: {tumor_likelihood}%")
print("Higher percentage suggests a higher chance of tumor presence.")
print("Note: This is just a model prediction, consult a doctor for diagnosis.")