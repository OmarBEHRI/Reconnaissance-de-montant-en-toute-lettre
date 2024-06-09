import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import sys
import io
import json
import csv

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# Define the path to your CSV file
csv_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "english.csv")

# Initialize an empty list to store the dataset
dataset, vocab = [], set()

# Open the CSV file and read its contents
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Skip the header row
    next(csv_reader)

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "letters_dataset")

    # Iterate over each row in the CSV
    for row in csv_reader:
        # Each row is in the format [image_path, label]
        image_path = os.path.join(dataset_path, row[0][4:])
        label = row[1]
        
        # Append the [image_path, label] pair to the dataset list
        dataset.append([image_path, label])
        vocab.update(list(label))


# Initialize lists to hold image data and labels
images = []
labels = []

# Loop through the dataset and process each image and label
for item in tqdm(dataset):
    image_path, label = item
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to the required size
    image = cv2.resize(image, (128, 96))
    # Convert the image to array
    image = img_to_array(image)
    # Append the image and label to the respective lists
    images.append(image)
    labels.append(label)

# Convert lists to numpy arrays
images = np.array(images, dtype="float") / 255.0  # Normalize pixel values
labels = np.array(labels)

# Encode labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.2, random_state=42)


model = Sequential()

# Add layers to the model
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(96, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

# Print the model summary
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

try:
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=25, batch_size=32)
except UnicodeEncodeError as e:
    # Handle the exception and print details
    error_message = f"UnicodeEncodeError: {e}"
    print(error_message, file=sys.stderr)

# Save history to a JSON file
history_dict = history.history
with open('training_history_lettres_model.json', 'w') as f:
    json.dump(history_dict, f)

loss, accuracy = model.evaluate(testX, testY)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("reconnaisance_de_lettres_model.h5")