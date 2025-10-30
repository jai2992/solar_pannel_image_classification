import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import DenseNet121
import numpy as np
from PIL import Image

# Define model parameters
img_height, img_width = 224, 224
num_classes = 6  # Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered

@st.cache_resource
def load_model():
    """Rebuild the model architecture and load weights"""
    # Rebuild the model architecture from scratch
    base_model = DenseNet121(
        include_top=False,
        weights=None,  # Don't load ImageNet weights
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False
    
    # Create the same model architecture used during training
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Load weights from the saved model
    model.load_weights("model_densenet.h5")
    
    return model

# Load the model
new_model = load_model()

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

    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)


    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]


    st.success(f"### Prediction: **{predicted_class}**")


