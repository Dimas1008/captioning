import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('/content/caption_model.h5')

st.title("Captioning Penyakit Ulkus Diabetes")

uploaded_file = st.file_uploader("Pilih gambar...", type="jpg")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])  # Adjust the relative width of the columns

    with col1:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        st.image(img, caption='Uploaded Image', width=300)

    with col2:
        st.write("")
        st.write("Classifying...")

        prediction = model.predict(img_array)
        probability = prediction[0][0]

        if probability > 0.5 :
            st.write("Prediksi: Batu Ginjal")
            st.write(f"Probabilitas: {probability*100:.2f}%")
        else:
            st.write("Prediksi: Normal")
            st.write(f"Probabilitas: {(1 - probability)*100:.2f}%")
