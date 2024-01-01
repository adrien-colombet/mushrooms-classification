import streamlit as st
import requests
from PIL import Image
import io

def app():
    st.title("Mushroom identification")
    st.header("Problem statement")
    st.text_area("the purpose of this application is to provide computer vision support to determine the edibility and classify of a mushroom according to a single photo taken with a smartphone. The app will suggest the order of the species and can provide further detail up to the genus.")

    # Image URL
    image_url = 'https://en.wikipedia.org/wiki/Mushroom#/media/File:Mushroom_cap_morphology2.png'

    # Download the image
    response = requests.get(image_url)
    morphology_image = Image.open(io.BytesIO(response.content))

    # Display the image
    st.image(morphology_image, caption='Mushroom Cap Morphology')