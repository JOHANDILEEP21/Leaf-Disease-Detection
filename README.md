# Leaf-Disease-Detection

## Overview

This project aims to develop a system for detecting diseases in plant leaves using advanced AI algorithms. By leveraging machine learning techniques, this system can accurately identify various plant diseases, helping farmers and agriculturists manage crop health more effectively.

## Features

- **Accurate Disease Detection**: Utilizes state-of-the-art machine learning models to identify plant diseases.
- **User-Friendly Interface**: Easy-to-use interface for uploading leaf images and receiving diagnostic results.
- **Comprehensive Database**: Extensive dataset of plant leaves with various diseases for training and validation.
- **Real-Time Processing**: Fast and efficient processing of images to provide real-time results.

## Technologies Used

- **Programming Languages**: Python
- **Libraries**: Numpy, Keras, Sckit learn, Mathplotlib
- **Tools**: Jupyter Notebook for experimentation and visualization (if using Python for training)

**Download the dataset:**

    Download the dataset from [Kaggle]([https://www.kaggle.com](https://www.kaggle.com/datasets/dev523/leaf-disease-detection-dataset)) and place it in the `data` directory.

## Usage

1. **Upload Image**: Upload an image of a plant leaf through the web interface.
2. **Processing**: The system processes the image and runs it through the trained machine learning model.
3. **Results**: The system displays the disease detected in the leaf, if any, along with confidence scores.

## Dataset

The dataset used in this project contains images of healthy and diseased plant leaves. Each image is labeled with the type of disease. The dataset is split into training, validation, and test sets.

## Model Training

1. **Data Preprocessing**: Images are resized, normalized, and augmented to enhance model performance.
2. **Model Architecture**: A Convolutional Neural Network (CNN) is used for image classification.
3. **Training**: The model is trained on the processed dataset using TensorFlow/Keras.
4. **Evaluation**: Model performance is evaluated on the validation set and fine-tuned for optimal accuracy.

## Proposed Model

### Dataset
- Plant village dataset with tomato samples, including six disorders.

### Classification
1. **Convolutional Neural Network (CNN)**:
    - Deep learning model that automatically extracts features and classifies the images.
  
### Evaluation:
    - Accuracy

### Deployed Link:

https://leaf-disease-detection-bjkkwbadsyk7f7pnyscctc.streamlit.app/

The model works well in my local machine. But not in streamlit app.


