import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load the model
model = load_model('./model_mobile_checkpoint')

# Create a file uploader to upload your images
uploaded_file = st.file_uploader("Choose a jpg image...", type="jpg")

# Classes
class_names = {0: "Agaricales", 1:"Auriculariales", 2:"Boletales", 3:"Cantharellales", 4:"Gomphales", 5:"Pezizales", 6:"Polyporales", 7:"Russulales"}

if uploaded_file is not None:
    # Open the image
    #image = Image.open(uploaded_file)
    img = keras.utils.load_img(uploaded_file, target_size=(224,224,3))
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image here if needed
    # For example, you might want to resize the image
    # image = image.resize((width, height))
    
    # Convert the image to a numpy array
    image = np.array(img)
    image = np.expand_dims(image, axis=0)
    
    # Make predictions
    predictions = model.predict(preprocess_input(image))
    st.write('Predicted class:', class_names[np.argmax(predictions)])
    st.write(predictions)
