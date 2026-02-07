import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page title
st.set_page_config(page_title="NutriGaze - Smart Sorting", layout="centered")

st.title("🥗 NutriGaze – Smart Sorting System")
st.write("Upload a fruit or vegetable image to detect **Healthy or Rotten**")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("healthy_vs_rotten.h5")

model = load_model()

# Class labels (change if needed)
classes = [
    'Apple_Healthy', 'Apple_Rotten',
    'Banana_Healthy', 'Banana_Rotten',
    'Potato_Healthy', 'Potato_Rotten',
    'Tomato_Healthy', 'Tomato_Rotten'
]

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        result = classes[np.argmax(prediction)]

        st.success(f"✅ Prediction: **{result}**")
