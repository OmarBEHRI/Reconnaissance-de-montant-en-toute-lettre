import os
import csv

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

# Print the dataset to verify
print(dataset)

