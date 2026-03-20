import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("pokemon_model.h5")

# Class labels
class_names = ['bulbasaur', 'charmander', 'pikachu', 'squirtle']

# Stats
pokemon_stats = {
    "pikachu": {"HP": 35, "Attack": 55, "Defense": 40, "Speed": 90, "Type": "Electric"},
    "bulbasaur": {"HP": 45, "Attack": 49, "Defense": 49, "Speed": 45, "Type": "Grass/Poison"},
    "charmander": {"HP": 39, "Attack": 52, "Defense": 43, "Speed": 65, "Type": "Fire"},
    "squirtle": {"HP": 44, "Attack": 48, "Defense": 65, "Speed": 43, "Type": "Water"}
}

st.title("Pokémon Pokédex AI")

# Upload button
uploaded_file = st.file_uploader(" Upload a Pokémon image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((64,64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f" Prediction: {predicted_class.upper()}")

    # Show stats
    stats = pokemon_stats.get(predicted_class)

    if stats:
        st.subheader(" Pokémon Stats")
        for key, value in stats.items():
            st.write(f"**{key}:** {value}")
