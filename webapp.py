import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Rescale pixel values to [0, 1]
    return image

# Function to make predictions
def predict_class(image_path):
    image = preprocess_image(image_path)
    # Convert the image to a batch of size 1 (since we are predicting a single image)
    image_batch = np.expand_dims(image, axis=0)
    # Make predictions on the image
    predictions = model.predict(image_batch)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index

def main():
    st.title('Multi-Class Image Classification App')
    
    # File uploader
    image_file = st.file_uploader('Upload an image to check if it is a Car, Bike, Truck, Cart, Bicycle', type=['jpg', 'jpeg', 'png'])
    
    if image_file is not None:
        try:
            # Save the uploaded image to a temporary file
            temp_image_path = 'temp_image.jpg'
            with open(temp_image_path, 'wb') as f:
                f.write(image_file.getvalue())
            
            # Make prediction
            predicted_class_index = predict_class(temp_image_path)

            # Assuming you have a list of class names
            class_names = ['Bicycle', 'Bike', 'Car', 'Cart', 'Truck']
            predicted_class_name = class_names[predicted_class_index]

            # Display prediction
            st.image(image_file, caption='Uploaded Image', use_column_width=True)
            st.write(f'Prediction: {predicted_class_name}')
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.write('Please upload an image.')

if __name__ == '__main__':
    main()
