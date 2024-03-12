import traceback
from flask import Flask, request, render_template, redirect, send_file, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sys
import cv2
sys.stdout.reconfigure(encoding='utf-8')
# Load the trained model
model = load_model('brain.keras')

# Define function to preprocess an image
def preprocess_image(image_file):
    """
    Preprocesses an image for the model.
    """
    img = Image.open(image_file)
    img = img.resize((64, 64))  # Match your model's input shape
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return img

# Initialize Flask app
app = Flask(__name__)

# Set up static folder for potential assets (optional)
app.static_folder = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_predict():
    if 'image' not in request.files:
        print('No file uploaded')
        return redirect(request.url)

    image_file = request.files['image']
    if image_file.filename == '':
        print('No selected file')
        return redirect(request.url)

    try:
        uploaded_image_path = 'static/uploaded_image.jpg'
        image_file.save(uploaded_image_path)
        # Preprocess the image
        image = preprocess_image(image_file)
        if image is None:
            return "Error: Failed to preprocess the image"

        # Make prediction
        prediction = model.predict(image)
        class_index = np.argmax(prediction[0])
        prediction_text = "Tumor" if class_index == 1 else "No Tumor"

        # Return prediction result to the template
        return render_template('index.html', prediction=prediction_text, uploaded_image='/static/uploaded_image.jpg')

    except Exception as e:
        print(f'Error: {e}')
        return redirect(request.url)

# Define route for performing image segmentation
@app.route('/perform_segmentation', methods=['POST'])
def perform_segmentation():
    # Load the uploaded image
    uploaded_image_path = 'static/uploaded_image.jpg'

    try:
        # Perform segmentation and highlight tumor
        highlighted_image = highlight_tumor(uploaded_image_path, model)

        # Check the dimensions and content of the highlighted image
        print("Highlighted image shape:", highlighted_image.shape)
        print("Highlighted image content:", highlighted_image)

        # Save and display the segmented image
        save_and_display_image(highlighted_image)

        return "Segmentation completed"  # or redirect to another page

    except Exception as e:
        print(f'Error in segmentation: {e}')
        return "Error in segmentation"

def highlight_tumor(image_path, model):
    """
    Highlights the predicted tumor region in red in the input image.

    Args:
        image_path: The file path of the input image.
        model: The pre-trained image segmentation model.

    Returns:
        The image with the highlighted tumor region in red as a NumPy array.
    """

    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Match your model's input shape
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add a batch dimension

    # Perform segmentation
    prediction = model.predict(img)
    processed_prediction = refine_prediction(prediction)

    # Overlay mask on the original image
    mask = processed_prediction[0]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_image = cv2.imread(image_path)
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # Red color for contours

    return highlighted_image

def refine_prediction(prediction):
    """
    Refines the raw prediction mask (if needed).

    Args:
        prediction: The raw prediction mask from the model.

    Returns:
        The refined prediction mask.
    """
    # Placeholder for refinement logic
    return prediction

def save_and_display_image(image, filename="segmentation_image.jpg"):
    """
    Saves the image to a file and displays it on the screen.

    Args:
        image: The image to be saved and displayed.
        filename: The desired filename (default: "segmentation_image.jpg").
    """
    # Check if the image is empty or None
    if image is None or len(image) == 0:
        print("Error: Empty or invalid image")
        return

    # Attempt to write the image to the file
    try:
        success = cv2.imwrite(filename, image)
        if success:
            print("Image saved successfully.")
        else:
            print("Error: Failed to save image.")
    except Exception as e:
        print(f"Error while writing image: {e}")

    # Display the image
    cv2.imshow("Segmented Image", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
