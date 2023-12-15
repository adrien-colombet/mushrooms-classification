import streamlit as st
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import supervision as sv
import imageio
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def app():

    # download segment anything model
    url_sam = '1l99dTkaKmha8i3Qpq2UxPwFiJoNhLBUo'
    dest_path = './model/sam_vit_h_4b8939.pth'

    if not os.path.exists(dest_path):
        gdd.download_file_from_google_drive(file_id = url_sam,
                                        dest_path = dest_path,
                                        unzip = False)

    # instantiating segmentation
    sam = sam_model_registry["default"](checkpoint="./model/sam_vit_h_4b8939.pth").to()
    mask_generator = SamAutomaticMaskGenerator(sam)
    

    # Create a file uploader to upload your images
    uploaded_file = st.file_uploader("Choose a jpg image...", type="jpg")

    if uploaded_file is not None:
        # Open the image
        #image = Image.open(uploaded_file)
        img = tf.keras.utils.load_img(uploaded_file, target_size=(224,224,3))
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        
        st.image(img, caption='Uploaded Image.', use_column_width=True)    

        # generating masks
        array_img = imageio.v3.imread(uploaded_file)
        masks = mask_generator.generate(array_img)

        # annotation
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(masks)
        annotated_image = mask_annotator.annotate(array_img, detections)
        st.image(annotated_image, caption='Annotated Image.', use_column_width=True) 

        # Use the function
        selected_mask = select_mask(array_img, masks)
        selected_detect = sv.Detections.from_sam(selected_mask)
        selected_image = mask_annotator.annotate(array_img, selected_detect)
        st.image(selected_image, caption='Selecting main central mask', use_column_width=True) 
        


import numpy as np

def select_mask(array_img, masks):
    # Initialize an empty list to store the areas, colors, and centers of each mask
    areas = []
    colors = []
    centers = []

    # Loop over each mask
    for mask in masks:
        # Get the area of the mask
        area = mask['area']
        areas.append(area)

        # Calculate the average color of the mask
        color = np.mean(array_img[mask['segmentation']], axis=0)
        colors.append(color)

        # Calculate the center of the mask using the 'segmentation' key
        segmentation = mask['segmentation']
        x, y = np.where(segmentation)
        center = [np.mean(x), np.mean(y)]
        centers.append(center)

    # Convert the lists to numpy arrays for easier manipulation
    areas = np.array(areas)
    colors = np.array(colors)
    centers = np.array(centers)

    # Define the RGB values for black
    black = np.array([0, 0, 0])

    # Calculate the Euclidean distance from each color to black
    dist_to_black = np.linalg.norm(colors - black, axis=1)

    # Define a threshold for what counts as "close" to black
    threshold = 50

    # Find the masks that are not close to black
    non_black_mask_indices = dist_to_black > threshold

    # Calculate the center of the image
    image_center = np.array(array_img.shape[0:2]) / 2

    # Among these, find the indices of the three with the largest area
    largest_non_black_mask_indices = np.argsort(areas)[-3:]

    # Calculate the distance from each mask center to the image center
    dist_to_image_center = np.linalg.norm(centers[largest_non_black_mask_indices] - image_center, axis = 1)

    # Find the mask that is closest to the image center
    most_central_mask_index = np.argmin(dist_to_image_center)

    # Return this mask
    return [masks[largest_non_black_mask_indices[most_central_mask_index]]]