import os
import numpy as np
import cv2
import tensorflow as tf
from django.conf import settings
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, "brain_tumor_model2.h5")
model = load_model(MODEL_PATH)

# Get model input size
IMG_SIZE = model.input_shape[1]  # Automatically fetch the correct input size

# Ensure model architecture is correct
print("Model Summary:")
print(model.summary())

# Image Preprocessing Function
def preprocess_image(image_path):
    """
    Load and preprocess the image to match the model's expected input.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Ensure 3-channel (RGB) input
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values (0 to 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
    
    print(f"Preprocessed image shape: {img.shape}")  # Debugging

    return img

# Prediction Function
def predict_tumor(image_path):
    """
    Predicts the type of brain tumor from an MRI image.
    """
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Get probability scores
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get highest probability class
    
    # Ensure class labels match training labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    tumor_class = class_labels[predicted_class]  # Map index to class name
    
    print(f"Prediction Output: {prediction}")  # Debugging
    print(f"Predicted Class: {tumor_class}")  # Debugging
    
    return tumor_class
