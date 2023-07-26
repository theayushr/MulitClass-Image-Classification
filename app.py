from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf

# loading the trained model
model = tf.keras.models.load_model('my_model.keras')  # C:\Users\Ayush Rawat\Documents\CSI\CSI Project\

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# function to preprocess imgae
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Rescale pixel values to [0, 1]
    return image


# funtion to make predictions
def predict_class(image_path):
    image = preprocess_image(image_path)
    image_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(image_batch)
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index


@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    image_path = 'two.jpg'  
    image_file.save(image_path)

    predicted_class_index = predict_class(image_path)

    # Assuming you have a list of class names
    class_names = ['Bicycle', 'Bike', 'Car', 'Cart', 'Truck']
    predicted_class_name = class_names[predicted_class_index]

    return jsonify({'prediction': predicted_class_name})


