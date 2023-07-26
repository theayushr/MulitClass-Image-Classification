# Machine Learning Multi-Class Image Classification Project

This repository contains a multi-class image classification project using Convolutional Neural Networks (CNNs) to classify images of different objects, including bicycles, bikes, cars, carts, and trucks. The project involves data preprocessing, model building using transfer learning with MobileNet, model training, and deployment using Flask as a web application.

# Project Overview
The main goal of this project is to build a deep learning model that can accurately classify images into one of the five classes mentioned above. The dataset used in this project is collected from various sources and has been preprocessed to ensure consistency and compatibility.

# Project Structure
The project is organized as follows:

data: This directory contains the processed training, validation, and testing datasets. The images are divided into subfolders based on their respective classes.

model: This directory contains the trained model stored in the .keras format.

app.py: This is the Flask web application file that serves as the interface for the trained model.

templates: This directory contains the HTML template for the web application.

# Step-by-Step Process

Data Collection: Gather a diverse and well-labeled dataset of images for the five classes - bicycles, bikes, cars, carts, and trucks. Organize the data into the appropriate subfolders for training, validation, and testing datasets.

Data Preprocessing: Resize all the images to a uniform size, e.g., 224x224, and apply data augmentation techniques like rotation, flipping, and zooming to increase the dataset's diversity.

Model Building: Use the MobileNet architecture as a pre-trained model for transfer learning. Remove the top layers and add a Global Average Pooling layer followed by Dense layers with ReLU activation. The output layer should have 5 units with a softmax activation function to produce class probabilities.

Model Training: Compile the model using the Adam optimizer and categorical cross-entropy loss function. Train the model on the training dataset using the ImageDataGenerator and validation dataset using the same generator.

Model Evaluation: Evaluate the model's performance on the testing dataset to calculate metrics like accuracy, precision, recall, and F1 score.

Web Application: Create a Flask web application to serve as the interface for the trained model. The application will allow users to upload an image and receive a prediction of the class it belongs to.

Deployment: Deploy the Flask application on a server or cloud platform, making it accessible to users.

# Dependencies
To run the project, make sure you have the following dependencies installed:

Python 3.x
TensorFlow
Keras
Flask
NumPy
OpenCV
Matplotlib

# Conclusion
This project demonstrates the end-to-end process of building an image classification model using deep learning techniques and deploying it as a web application using Flask. The trained model can accurately predict the class of objects in input images, making it a useful tool for various real-world applications.
