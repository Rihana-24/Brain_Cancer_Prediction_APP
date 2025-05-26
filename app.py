import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from models.cnn_pytorch import get_pretrained_model  # Make sure this exists

# Set up folders
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
model_path_tf = 'Rihanatou_BANKOLE_model.h5'
model_path_pth = 'Rihanatou_BANKOLE_model.torch'

# TensorFlow model
if not os.path.exists(model_path_tf):
    st.error(f"The file {model_path_tf} was not found.")
model_tf = tf.keras.models.load_model(model_path_tf)

# PyTorch model
model_pth = get_pretrained_model()
if not os.path.exists(model_path_pth):
    st.error(f"The file {model_path_pth} was not found.")
model_pth.load_state_dict(torch.load(model_path_pth, map_location=torch.device('cpu')))
model_pth.eval()

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocessing
def preprocess_tf(img):
    img = img.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_pt(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = img.convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# --- Streamlit App ---
st.set_page_config(page_title="Brain Cancer Classification", layout="centered")
st.title("🧠 Brain Cancer Classification")

framework = st.radio("Select Framework", ("TensorFlow", "PyTorch"))

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    if st.button("Predict"):
        if framework == "TensorFlow":
            input_img = preprocess_tf(image)
            preds = model_tf.predict(input_img)
            pred_class = class_names[np.argmax(preds[0])]
        else:
            input_img = preprocess_pt(image)
            with torch.no_grad():
                output = model_pth(input_img)
                _, predicted = torch.max(output, 1)
                pred_class = class_names[predicted.item()]

        st.success(f"✅ Predicted Class: **{pred_class.upper()}**")
