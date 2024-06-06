import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Define the labels (CIFAR-10)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Set up the Streamlit app
st.title('Image Classification with CNN')

st.header('Upload an Image for Classification')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((32, 32))  # Resize to the model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Display the prediction
    st.write(f'Prediction: {class_names[predicted_class]}')
