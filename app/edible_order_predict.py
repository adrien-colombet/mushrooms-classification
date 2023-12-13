import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
from google_drive_downloader import GoogleDriveDownloader as gdd
import os

def app():
    # downloading models
    url_order_model = '1KNN8xDcEVd0fMSp0JfgNaBUqOlumxaQY'
    dest_path = './model/MobileNetModel.hdf5'

    if not os.path.exists(dest_path):
        gdd.download_file_from_google_drive(file_id=url_order_model,
                                        dest_path=dest_path,
                                        unzip=False)

    url_edibility_model = '1PLWvsMlDTy-dQkL_Jahxh3RHqbhWBmKu'
    dest_path = './model/VF_ResNet50V2.hdf5'

    if not os.path.exists(dest_path):
        gdd.download_file_from_google_drive(file_id=url_edibility_model,
                                        dest_path=dest_path,
                                        unzip=False)

    # Load the model
    order_model = load_model('./model/MobileNetModel.hdf5')

    edbility_model = load_model('./model/VF_ResNet50V2.hdf5')

    # Create a file uploader to upload your images
    uploaded_file = st.file_uploader("Choose a jpg image...", type="jpg")

    # Classes
    edible_class_names = {0: "Non comestible", 1:"Comestible"}
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
        
        # Prediction on edibility
        edbility_predictions = edbility_model.predict(resnet_preprocess_input(image))
        st.write('Edibility:', edible_class_names[np.argmax(edbility_predictions)])
        st.write(edbility_predictions)

        # Make predictions
        predictions = order_model.predict(mobile_preprocess_input(image))
        st.write('Predicted order:', class_names[np.argmax(predictions)])
        st.write(predictions)

        


