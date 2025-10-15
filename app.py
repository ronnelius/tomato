# app.py
import streamlit as st
import cv2
import numpy as np
from detect import detect_objects
import tempfile

st.set_page_config(page_title="WebApp", layout="centered")
st.title("Tomatoes Object Detection WebApp")

# Upload model and image
model_path = st.text_input("Enter model", "my_model.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

    if st.button("Run Detection"):
        output = detect_objects(image.copy(), model_path)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)


st.write("Or use webcam:")
use_webcam = st.checkbox("Enable webcam")

if use_webcam:
    picture = st.camera_input("Take a picture")
    if picture:
        bytes_data = picture.getvalue()
        np_arr = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        output = detect_objects(frame, model_path)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Webcam Detection")





