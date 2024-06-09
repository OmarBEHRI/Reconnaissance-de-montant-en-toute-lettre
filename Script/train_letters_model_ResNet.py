import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Dropout, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import sys
import io
import os
import json
import csv
import cv2

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

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def residual_block(x, filters, kernel_size=3, stride=1):
    # First convolution
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    # Second convolution
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization()(y)

    # Adding the residual
    if stride != 1 or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)

    out = Add()([x, y])
    out = LeakyReLU()(out)

    return out

def build_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(16, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Residual blocks
    x = residual_block(x, 16)
    x = Dropout(0.3)(x)
    x = residual_block(x, 16, stride=2)
    x = Dropout(0.3)(x)
    x = residual_block(x, 32)
    x = Dropout(0.3)(x)
    x = residual_block(x, 32, stride=2)
    x = Dropout(0.3)(x)
    x = residual_block(x, 64)
    x = Dropout(0.3)(x)
    x = residual_block(x, 64, stride=2)
    x = Dropout(0.3)(x)
    x = residual_block(x, 64)
    x = Dropout(0.3)(x)
    x = residual_block(x, 64)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

input_shape = (96, 128, 3)  # Height, Width, Channels
num_classes = 62

model = build_resnet(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")

try:
    history = model.fit(
        datagen.flow(trainX, trainY, batch_size=32),
        validation_data=(testX, testY),
        epochs=100,
        callbacks=[reduceLROnPlat]
    )
except UnicodeEncodeError as e:
    # Handle the exception and print details
    error_message = f"UnicodeEncodeError: {e}"
    print(error_message, file=sys.stderr)

# Save history to a JSON file
history_dict = history.history
with open('training_history_lettres_model_ResNet.json', 'w') as f:
    json.dump(history_dict, f)

loss, accuracy = model.evaluate(testX, testY)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("reconnaisance_de_lettres_model_with_augmentation_ResNet.h5")
