import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class LeafDiseaseDetection:
    @staticmethod
    def load_model1():
        try:
            model = load_model('model.keras')
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    @staticmethod
    def predict_image(model, path):
        try:
            test_image = image.load_img(path, target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            return result
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None

    @staticmethod
    def testing(path):
        model = LeafDiseaseDetection.load_model1()
        if model is None:
            return
        result = LeafDiseaseDetection.predict_image(model, path)
        if result is None:
            return

        labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                  'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                  'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
                  'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                  'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                  'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                  'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
                  'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot',
                  'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                  'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                  'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

        predicted_label = labels[result.argmax()]
        probability = np.max(result)
        
        if 'healthy' in predicted_label:
            st.success(f"Your leaf has no disease. It is: {predicted_label} with a probability of {probability:.2f}")
        else:
            st.warning(f"Your leaf disease is: {predicted_label} with a probability of {probability:.2f}")

# Streamlit app
def main():
    st.title("Leaf Disease Detection")

    # File uploader
    upload_file = st.file_uploader("Choose a leaf image...", type="jpg")

    if upload_file is not None:
        st.image(upload_file, caption='Uploaded Image.', width=240)
        LeafDiseaseDetection.testing(upload_file)

if __name__ == "__main__":
    main()
