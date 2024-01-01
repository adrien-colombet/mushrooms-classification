import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import preprocess_input as mobile_preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input

from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import matplotlib.cm as cm

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

    # Load the models
    problem_names = ["edibility", "order", "mixed"]

    edbility_model_names = ["VF_ResNet50V2", "TL_ResNet50V2", "TL_VGG19", "TL_Xception", "TL_InceptionV3"]
    edbility_model_list = [load_model('./model/' + model_name + '.hdf5') for model_name in edbility_model_names]
    edbility_preprocess_list = [resnet_preprocess_input, resnet_preprocess_input, vgg19_preprocess_input, xception_preprocess_input, inception_preprocess_input]

    order_model_names = ["MobileNetModel"]
    order_model_list = [load_model('./model/' + model_name + '.hdf5') for model_name in order_model_names]
    order_preprocess_list = [mobile_preprocess_input]

    mixed_model_names = ["model_mobile_mixed"]
    mixed_model_list = [load_model('./model/' + model_name + '.hdf5') for model_name in mixed_model_names]
    mixed_preprocess_list = [mobile_preprocess_input]

    model_names_list = [edbility_model_names, order_model_names, mixed_model_names]
    models_list = [edbility_model_list, order_model_list, mixed_model_list]
    preprocess_list = [edbility_preprocess_list, order_preprocess_list, mixed_preprocess_list]

    # Create a file uploader to upload your images
    uploaded_file = st.file_uploader("Choose a jpg image...", type="jpg")

    # Classes
    edible_class_names = {0: "Non comestible", 1:"Comestible"}
    order_class_names = {0: "Agaricales", 1:"Auriculariales", 2:"Boletales", 3:"Cantharellales", 4:"Gomphales", 5:"Pezizales", 6:"Polyporales", 7:"Russulales"}
    mixed_class_names = {0: 'Agaricus', 1: 'Amanitaceae', 2: 'Auriculariales', 3: 'Boletus', 4: 'Butyriboletus', 5: 'Calvatia', 6: 'Cantharellales', 7: 'Clitocybe', 8: 'Coprinus', 9: 'Cortinariaceae', 10: 'Entolomataceae', 11: 'Fistulinaceae', 12: 'Gomphales', 13: 'Gomphidiaceae', 14: 'Gyroporaceae', 15: 'Hydnangiaceae', 16: 'Hygrophoraceae', 17: 'Hygrophoropsidaceae', 18: 'Imleria', 19: 'Infundibulicybe', 20: 'Leccinum', 21: 'Lepista', 22: 'Leucoagaricus', 23: 'Lyophyllaceae', 24: 'Macrolepiota', 25: 'Marasmiaceae', 26: 'Neoboletus', 27: 'Omphalotaceae', 28: 'Pezizales', 29: 'Physalacriaceae', 30: 'Pleurotus', 31: 'Pluteaceae', 32: 'Polyporales', 33: 'Russulales', 34: 'Strophariaceae', 35: 'Suillaceae', 36: 'Suillellus', 37: 'Tricholoma'}

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
        
        
        preds_list = [None]*3
        selected_model = [None]*3
        model = [None]*3
        preprocess = [None]*3

        for idx, (pb_name, model_names, models, preprocess_functions, col) in enumerate( zip(problem_names, model_names_list, models_list, preprocess_list, st.columns(3))):
            with col:
                # Selecting model
                selected_model[idx] = st.selectbox(
                    f'Model for problem {pb_name}',
                    options=model_names,
                    index=0)
                
                sel = model_names.index(selected_model[idx])

                model[idx] = models[sel]
                preprocess[idx] = preprocess_functions[sel]
                    
                # Making model predictions
                preds_list[idx] = model[idx].predict(preprocess[idx](image))

        # Generate class activation heatmap 
        selected_class = [None]*3

        for idx, ( preds, class_names, col) in enumerate(zip([preds_list[0], preds_list[1], preds_list[2]],  [edible_class_names, order_class_names, mixed_class_names], st.columns(3))):
            with col:

                st.write(problem_names[idx] + ':', class_names[np.argmax(preds)])
                st.write(preds)
                
                # Create a select box with the indexes of the array
                selected_class[idx] = st.selectbox(
                    f'Class index for model {problem_names[idx]}',
                    options=[i for i in range(preds.shape[1])],
                    index=int(np.argmax(preds)))
                
                last_conv_layer_name = None
                #for layer in model.layers:
                #    if isinstance(layer, tf.keras.layers.Conv2D):
                #        last_conv_layer_name = layer.name

                for layer in (model[idx].layers):
                    if len(layer.output_shape) == 4:
                        last_conv_layer_name = layer.name

                heatmap = make_gradcam_heatmap(preprocess[idx](image), model[idx], last_conv_layer_name, preds, selected_class[idx])
                st.write(last_conv_layer_name)
                display_gradcam(array, heatmap, alpha=0.4)

        


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, preds, pred_index=None):


    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        inputs = model.inputs, 
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds)
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Display Grad CAM
    st.image(superimposed_img, caption='Gradcam Image.', use_column_width=True)   