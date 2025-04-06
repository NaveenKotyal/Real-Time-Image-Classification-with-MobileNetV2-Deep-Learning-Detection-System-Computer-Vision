import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions,MobileNetV2


# Load model
model = MobileNetV2(weights='imagenet')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)[0]  # Only if model is ImageNet-based


    # Display predictions
    for i, (img_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}. {label}: {score*100:.2f}%")

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image to classify it using a pre-trained model.")

img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img is not None:
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Save to temp file
    with open("temp_image.jpg", "wb") as f:
        f.write(img.getbuffer())

    predict_image("temp_image.jpg")
    st.write("Done!")
