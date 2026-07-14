import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# The 10 CheXpert target labels in the exact order outputted by the model
LABELS = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Support Devices'
]

@st.cache_resource
def load_medical_model():
    """
    Load the trained CheXpert classification model.
    Uses @st.cache_resource to cache the model in memory.
    Checks for both 'best_model_finetuned.keras' and 'best_model.h5'.
    """
    model_paths = ['best_model_finetuned.keras', 'best_model.h5']
    selected_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            selected_path = path
            break
            
    if selected_path is None:
        raise FileNotFoundError(
            f"No pre-trained model found. Checked: {', '.join(model_paths)}. "
            "Please ensure your trained model is placed in the project root."
        )
    
    # We use compile=False during load because we only need the model for inference.
    # This bypasses compilation, preventing dependency issues with custom loss functions
    # (e.g. weighted cross-entropy) commonly used in multi-label chest X-ray models.
    try:
        model = load_model(selected_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {selected_path} due to error: {e}")

def preprocess_image(image_file, target_size=(224, 224)):
    """
    Preprocess the uploaded chest X-ray image.
    - Loads the image using PIL.
    - Converts it to RGB (in case of 1-channel grayscale inputs, as DenseNet expects 3 channels).
    - Resizes to the target size (224x224).
    - Converts to a numpy array.
    - Expands batch dimensions to (1, height, width, channels).
    - Standardizes the image using DenseNet's preprocess_input (ImageNet scaling and normalization).
    """
    # Open the image file
    img = Image.open(image_file)
    
    # Ensure RGB color space
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize to the model's expected shape
    img = img.resize(target_size)
    
    # Convert image to float32 numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Expand dimensions to create batch shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply DenseNet specific standardization (scales values to [0,1] and standardizes with ImageNet mean/std)
    preprocessed_img = preprocess_input(img_array)
    
    return preprocessed_img
