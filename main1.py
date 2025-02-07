import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

class LeafDiseaseDetection:
    
    @staticmethod
    def testing(image_path):
        model_path = 'Leaf_disease_detection.pkl'
        
        try:
            with open(model_path, 'rb') as file:
                ldd_model = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            return None
            
        test_image = image.load_img(image_path, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        if ldd_model:
            result = ldd_model.predict(test_image)
        else:
            st.error("Model could not be loaded.")
            return None
        
        fresult = np.max(result)
        
        label = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
                 'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
                 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
                 'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot',
                 'Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                 'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                 'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
                 'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
                 'Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch',
                 'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight',
                 'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                 'Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        
        label2 = label[result.argmax()]

        if result is not None:
            st.success(f"Your leaf disease is: {label2}")
        else:
            st.warning("Failed to get a prediction due to model loading issues.")

# Streamlit app
def main():
    st.title("Leaf Disease Detection")

    # File uploader
    upload_file = st.file_uploader("Choose a leaf image...", type="jpg")

    if upload_file is not None:
        
        #resize_image = image.load_img(upload_image, target_size=(240, 320))
        st.image(upload_file, caption='Uploaded Image.', width= 240) #, use_column_width=True)

        LeafDiseaseDetection.testing(upload_file)        

if __name__ == "__main__":
    main()
