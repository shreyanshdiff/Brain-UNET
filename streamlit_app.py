import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
# import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = (256 , 256)
MODEL_PATH = "brain_tumor_unetV2.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize(IMG_SIZE)
    image = np.array(image)/255.0
    image = np.expand_dims(image , axis = 0)
    image = np.expand_dims(image , axis = -1)
    return image

st.title("MRI BRAIN TUMOR USING UNET")
st.write("Upload Scan")

uploaded_file = st.file_uploader("Choose scan" , type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image , caption="Uploaded MRI Scan" , use_column_width=True)
    preprocessed_image = preprocess_image(image)
    
    prediction = np.mean(model.predict(preprocessed_image))
      # Ensures a single scalar value
  # Ensure scalar value
    result = "Detected" if prediction > 0.5 else "Not Detected"
    
    st.subheader("Result")
    st.write(result)
    
    st.write(f"confidence:{prediction:.2f}")
