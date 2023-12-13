import streamlit as st
from PIL import Image
import os
from glob import glob
from streamlit_cropper import st_cropper

def app():
    # Set the path to the directory containing the subdirectories with images
    path = os.path.join('..','dataset','order','train')

    # Get list of subdirectories
    subdirectories = [f.name for f in os.scandir(path) if f.is_dir()]

    # Create a select box for the subdirectories
    selected_subdir = st.selectbox('Select a subdirectory', subdirectories)

    # Get list of .jpg images in the selected subdirectory
    image_files = [f for f in os.listdir(os.path.join(path, selected_subdir)) if f.endswith('.jpg')]

    # Create a select box for the .jpg images
    selected_image = st.selectbox('Select an image', image_files)

    filename = os.path.join(path, selected_subdir, selected_image)

    # Get a list of all JPG images in the subdirectories
    #images = glob(f'{path}/**/*.jpg', recursive=True)

    # Create a dictionary to map image filenames to their paths
    #image_dict = {os.path.splitext(os.path.basename(image))[0]: image for image in images}

    # Create a selectbox for the image filenames
    #filename = st.selectbox('Select image filename', list(image_dict.keys()))

    # Display the selected image
    if 1: #filename in image_dict:
        #image = Image.open(image_dict[filename])
        image = Image.open(filename)

        # Get a cropped image from the frontend
        aspect_ratio = (1,1 )
        box_color = '#0000FF'
        realtime_update = 'True'
        cropped_img = st_cropper(image, realtime_update=realtime_update, box_color=box_color,
                                    aspect_ratio=aspect_ratio)
        
        # Manipulate cropped image at will
        _ = cropped_img.thumbnail((300,300))

        # Create 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(cropped_img, "Cropped Image") 

        # Create a text box containing the name of the image prefixed with "crop_"+ n
        with col2:
            n = 1
            new_filename = selected_image[:-4] + f"_crop_{n}" + '.jpg'
            while os.path.exists(os.path.join(path, selected_subdir, new_filename + '.jpg')):
                n += 1
                new_filename = selected_image[:-4] + f"_crop_{n}" + '.jpg'
            st.text_input('New filename', new_filename)  
    
            # Add a button that saves the cropped image in the same directory with the new filename
            if st.button('Save Cropped Image'):
                cropped_img.save(os.path.join(os.path.join(path, selected_subdir, new_filename + '.jpg')))
                st.write('Image saved successfully.')
    else:
        st.write(f'Image not found: {filename}')
