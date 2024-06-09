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

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_dir = os.path.join(project_dir, "ProcessedImages")

dataset, vocab, maxlen = [], set(), 0

chiffres = open(os.path.join(project_dir, "chiffres.txt"), "r").readlines()

for line in tqdm(chiffres):
    line_split = line.split(" ")
    file_name = line_split[0]
    label = line_split[-1].rstrip("\n")
    image_path = os.path.join(dataset_dir, file_name)
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        continue
    dataset.append([image_path, label])
    vocab.update(list(label))
    maxlen = max(maxlen, len(label))


# Initialize lists to hold image data and labels
images = []
labels = []

# Loop through the dataset and process each image and label
for item in dataset:
    image_path, label = item
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image to the required size
    image = cv2.resize(image, (155, 50))
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
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(50, 155, 3)))
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
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

loss, accuracy = model.evaluate(testX, testY)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("reconnaisance_de_mot_model.h5")