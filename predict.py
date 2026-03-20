import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("pokemon_model.h5")

# Class labels (IMPORTANT: match dataset folders)
class_names = ['bulbasaur', 'charmander', 'pikachu', 'squirtle']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64,64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    print(f"Prediction: {predicted_class}")

# Test
predict_image("test.jpg")
