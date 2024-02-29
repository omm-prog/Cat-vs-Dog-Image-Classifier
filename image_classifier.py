# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

# Replace '/content/drive' with 'your data path'
your_data_path = 'your data path'
drive.mount(your_data_path)

# Setting up data generators for training and testing
train_data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_generator = ImageDataGenerator(rescale=1./255)

train_dataset = train_data_generator.flow_from_directory(f'{your_data_path}/MyDrive/data/dataset/training_data/', target_size=(64, 64), batch_size=32, class_mode='binary')
test_dataset = test_data_generator.flow_from_directory(f'{your_data_path}/MyDrive/data/dataset/testing_data/', target_size=(64, 64), batch_size=32, class_mode='binary')

# Building the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_dataset, steps_per_epoch=len(train_dataset), epochs=10, validation_data=test_dataset, validation_steps=len(test_dataset))

# Evaluating the model on the test dataset
loss, accuracy = model.evaluate(test_dataset, steps=len(test_dataset))
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Making predictions on a new image
from tensorflow.keras.preprocessing import image
import numpy as np

# Input image path from the user
img_path = input("Enter the path to the image you want to classify: ")

# Load and preprocess the image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
prediction = model.predict(img_array)

# Interpret the predictions
class_label = "Cat" if prediction < 0.5 else "Dog"
probability = prediction[0][0]

# Display the result
print(f"The image is predicted to be a {class_label} with probability {probability}")
