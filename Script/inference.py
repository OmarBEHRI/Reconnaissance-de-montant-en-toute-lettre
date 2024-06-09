import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import os
import sys
import io

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')



# Load the model
model = load_model("french_word_recognition_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (155, 50))
    image = img_to_array(image)
    image = image.astype("float") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make prediction
def make_prediction(model, image_path, label_encoder):
    preprocessed_image = preprocess_image(image_path)
    try:
        predictions = model.predict(preprocessed_image)
    except UnicodeEncodeError as e:
        # Handle the exception and print details
        error_message = f"UnicodeEncodeError: {e}"
        print(error_message, file=sys.stderr)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])
    return predicted_label[0]

# Recreate the label encoder with the original labels
original_labels = ["un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante", "cinquante", "soixante", "cent", "cents", "milles", "mille", "million", "millions", "milliard", "milliards", "-", "et", "virgule", "MAD", "dirhams"]
label_encoder = LabelEncoder()
label_encoder.fit(original_labels)

# Run inference
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define the directory path
directory_path = os.path.join(project_dir, "ProcessedImages")
# Get the list of all files in the directory
filenames = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
for name_file in filenames:
    image_path = dataset_dir = os.path.join(project_dir, "ProcessedImages", name_file)
    predicted_label = make_prediction(model, image_path, label_encoder)
    print(f"The predicted label for the image {name_file} is: {predicted_label}")
