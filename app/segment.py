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