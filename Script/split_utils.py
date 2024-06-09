import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

def split_dataset(dataset):
    test_data =[]
    training_data = []
    for i in range(len(dataset)):
        if i < 680:
            if (i%20) < 4:
                test_data.append(dataset[i])
            else:
                training_data.append(dataset[i])
        else:
            training_data.append(dataset[i])
    return training_data, test_data

def preprocess_data(dataset):
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
    return images, labels


class CTCloss(tf.keras.losses.Loss):
    """ CTCLoss objec for training the model"""
    def __init__(self, name: str = "CTCloss") -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Compute the training batch CTC loss value"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss