import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

new_model = tf.keras.models.load_model("model_densenet.h5")

class_names = [
    'Bird-drop', 'Clean', 'Dusty', 
    'Electrical-damage', 'Physical-Damage', 
    'Snow-Covered'
]

st.title("Solar Panel Image Classifier")
st.subheader("Upload an image to classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img_height, img_width = 224, 224
    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]


    st.success(f"### Prediction: **{predicted_class}**")


