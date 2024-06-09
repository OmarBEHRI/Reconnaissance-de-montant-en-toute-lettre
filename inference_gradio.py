import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import io
import sys

# Ensure UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# # Load your models
# letter_model = load_model('reconnaisance_de_lettres_model_with_augmentation.h5')
# word_model = load_model('reconnaisance_de_mots_augmentation_model.h5')

# List of characters from '0' to '9'
digits = [chr(i) for i in range(48, 58)]

# List of characters from 'a' to 'z'
lowercase_letters = [chr(i) for i in range(97, 123)]

# List of characters from 'A' to 'Z'
uppercase_letters = [chr(i) for i in range(65, 91)]

# Combine all the lists into one
all_characters = digits + lowercase_letters + uppercase_letters


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def split_table(image, rows, cols):
    h, w = image.shape[:2]
    cell_height = h // rows
    cell_width = w // cols
    cells = []
    for i in range(rows):
        for j in range(cols):
            cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
            cv2.imshow('Cell', cell)
            cells.append(cell)
            
    return cells

def process_image(image, resize_dims, crop_coords, cell_dims):
    # Resize the input image
    resized_image = resize_image(image, resize_dims[0], resize_dims[1])
    
    # Crop the resized image
    cropped_image = crop_image(resized_image, *crop_coords)
    
    # Split the cropped image into table cells
    cells = split_table(cropped_image, cell_dims[0], cell_dims[1])
    
    return cells

def run_inference(cells, model, cell_size, model_choice):
    results = []
    for cell in cells:
        # Resize cell image to the preferred input size for the model
        cell_resized = resize_image(cell, cell_size[0], cell_size[1])
        
        # Normalize the cell image as required by the model
        cell_resized = cell_resized / 255.0  # Assuming the model expects pixel values between 0 and 1
        cell_resized = np.expand_dims(cell_resized, axis=0)  # Adding batch dimension
        
        # Run inference
        try:
            result = model.predict(cell_resized)
        except UnicodeEncodeError as e:
            # Handle the exception and print details
            error_message = f"UnicodeEncodeError: {e}"
            print(error_message, file=sys.stderr)
        if model_choice == "Letter":
            character = all_characters[max(enumerate(result[0]), key=lambda x: x[1])[0]]
            results.append(character)
        else:
            result.append(result)
    return results

def gradio_interface(image, model_choice, width, height, x, y, w, h, rows, cols, cell_width, cell_height):
    # resize_dims = (width, height)
    # crop_coords = (x, y, w, h)
    # cell_dims = (rows, cols)
    
    # # Select the model based on user choice
    # model = letter_model if model_choice == 'Letter' else word_model
    
    # # Process the input image
    # cells = process_image(image, resize_dims, crop_coords, cell_dims)
    
    # # Run inference on the processed cells
    # results = run_inference(cells, model, (cell_width, cell_height), model_choice)
    
    return "deux cents mille dirhams"
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Dropdown(choices=['Letter', 'Word'], label="Model Choice"),
        gr.Number(label="Resize Width", value=800),
        gr.Number(label="Resize Height", value=600),
        gr.Number(label="Crop X Coordinate", value=0),
        gr.Number(label="Crop Y Coordinate", value=0),
        gr.Number(label="Crop Width", value=800),
        gr.Number(label="Crop Height", value=200),
        gr.Number(label="Number of Rows in Table", value=1),
        gr.Number(label="Number of Columns in Table", value=30),
        gr.Number(label="Cell Resize Width", value=28),
        gr.Number(label="Cell Resize Height", value=28)
    ],
    outputs=gr.JSON(label="Results")
)

iface.launch()