import sys
import os

# Add project root to sys.path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.model import PlantClassifier
from src.data_loader import val_transform, get_loaders  # Reuse transform and classes

# Load classes (from any loader; small batch to load quickly)
_, _, _, classes = get_loaders(batch_size=1)
num_classes = len(classes)

# Load model on CPU for portability (change to 'cuda' if GPU available locally)
device = torch.device('cpu')
model = PlantClassifier(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

# Custom CSS for amazing, modern UI: Dark background, green accents, glowing, soft edges, animations
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #121212;  /* Dark background */
        color: #E0E0E0;  /* Light text for contrast */
    }
    /* Header styling */
    .header {
        text-align: center;
        color: #27AE60;  /* Fancy green */
        font-size: 3em;
        margin-bottom: 20px;
        text-shadow: 0 0 10px #27AE60;  /* Glowing effect */
        animation: fadeIn 2s ease-in-out;
    }
    /* Upload area and buttons with soft edges, glow */
    .stFileUploader, .stButton > button {
        border-radius: 15px;  /* Soft edges */
        box-shadow: 0 0 15px #27AE60;  /* Green glow */
        background-color: #1E1E1E;  /* Dark card */
        color: #27AE60;
        border: 1px solid #27AE60;
        animation: pulse 2s infinite;  /* Alive pulse animation */
    }
    /* Prediction box: Glowing green card */
    .prediction-box {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px #27AE60;
        animation: fadeIn 1s ease-in-out;
        text-align: center;
    }
    /* Image preview with soft borders */
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 0 10px #27AE60;
        max-width: 100%;
    }
    /* Sidebar styling */
    .css-1lcbmhc {
        background-color: #1A1A1A;
        border-right: 1px solid #27AE60;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 #27AE60; }
        50% { box-shadow: 0 0 20px #27AE60; }
        100% { box-shadow: 0 0 0 #27AE60; }
    }
    /* Confetti JS for alive feel on prediction (using canvas-confetti library via CDN) */
</style>
""", unsafe_allow_html=True)

# Add confetti script for celebration on prediction
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
<script>
function triggerConfetti() {
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#27AE60', '#121212', '#E0E0E0']
    });
}
</script>
""", unsafe_allow_html=True)

# App layout
st.set_page_config(layout="wide", page_title="Plant Disease Classifier", page_icon="🌿")

# Glowing header
st.markdown('<div class="header">Plant Disease Classifier Demo</div>', unsafe_allow_html=True)

# Sidebar for info
with st.sidebar:
    st.markdown("### Model Info")
    st.write("Trained on PlantVillage dataset (38 classes).")
    st.write(f"Test Accuracy: 99.67%")
    st.markdown("### Classes")
    for c in classes:
        st.write(f"- {c}")
    st.markdown("### Upload Tips")
    st.write("Upload a clear leaf photo (JPG/PNG).")
    st.markdown("### GitHub")
    st.write("[View Code](https://github.com/Rayan-Aisami/plant-disease-image-classifier)")

# Main content: Upload and predict
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Upload Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            # Display uploaded image with soft borders/glow using markdown <img>
            image = Image.open(uploaded_file)
            st.markdown(f'<img src="data:image/png;base64,{image_to_base64(image)}" class="uploaded-image" alt="Uploaded Image">', unsafe_allow_html=True)
            st.caption("Uploaded Image")  # Caption below
        except Exception as e:
            st.error(f"Error loading image: {e}. Please upload a valid JPG/PNG.")

with col2:
    if uploaded_file is not None:
        st.markdown("### Prediction")
        with st.spinner('Analyzing image...'):  # Loading animation
            try:
                # Preprocess image
                img_array = np.array(image.convert('RGB'))
                transformed = val_transform(image=img_array)
                input_tensor = transformed['image'].unsqueeze(0).to(device)  # Batch dim

                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred_idx = output.argmax(1).item()
                    confidence = probs[pred_idx].item() * 100

                # Display in glowing box
                pred_class = classes[pred_idx]
                st.markdown(f'<div class="prediction-box">Predicted Disease: <strong>{pred_class}</strong><br>Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

                # Confetti celebration for alive feel
                st.markdown('<script>triggerConfetti();</script>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}. Try another image.")
    else:
        st.write("Upload an image to get started!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & PyTorch | © 2025 Ryan | For educational purposes.")

# Helper function for base64 image (for markdown <img>)
def image_to_base64(img):
    from io import BytesIO
    import base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()