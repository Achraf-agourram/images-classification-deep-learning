import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.prediction.predict import load_prediction_resources


@st.cache_resource
def get_prediction_resources():
    return load_prediction_resources()


st.set_page_config(page_title="Landscape image classifier", page_icon="🌄")
st.title("Landscape image classifier")
st.write("Upload an image to classify it as one of the model's landscape categories.")

try:
    model, class_names = get_prediction_resources()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_image = st.file_uploader(
    "Choose an image",
    type=("jpg", "jpeg", "png"),
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    image_tensor = tf.keras.utils.img_to_array(image)
    image_tensor = tf.image.resize(image_tensor, (150, 150))
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    probabilities = model.predict(
        tf.expand_dims(image_tensor, axis=0),
        verbose=0,
    )[0]

    if len(probabilities) != len(class_names):
        st.error("The number of model outputs does not match the class names.")
        st.stop()

    predicted_index = int(np.argmax(probabilities))
    st.subheader(f"Prediction: {class_names[predicted_index]}")
    st.metric("Confidence", f"{probabilities[predicted_index]:.2%}")

    class_probabilities = pd.DataFrame(
        {"Class": class_names, "Probability": probabilities}
    ).sort_values("Probability", ascending=False)
    st.dataframe(class_probabilities, hide_index=True, use_container_width=True)
