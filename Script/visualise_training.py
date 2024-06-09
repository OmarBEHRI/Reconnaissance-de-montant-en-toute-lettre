import json
import pickle
import matplotlib.pyplot as plt

# Load history from the JSON file
with open('training_history_lettres_model_ResNet.json', 'r') as f:
    history_dict = json.load(f)

# Plot the training and validation loss
plt.plot(history_dict['loss'], label='train_loss')
plt.plot(history_dict['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# If you also have accuracy metrics, you can plot them similarly
if 'accuracy' in history_dict:
    plt.plot(history_dict['accuracy'], label='train_accuracy')
    plt.plot(history_dict['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# # Load the training history
# with open("training_history_word_ResNet.pkl", 'rb') as f:
#     history = pickle.load(f)

# # Compute 1 - WER and 1 - val_WER
# training_accuracy = [1 - wer for wer in history['WER']]
# validation_accuracy = [1 - val_wer for val_wer in history['val_WER']]

# # Plot the accuracy evolution
# plt.figure(figsize=(10, 6))
# plt.plot(training_accuracy, label='Training Accuracy')
# plt.plot(validation_accuracy, label='Validation Accuracy')
# plt.title('Accuracy Evolution')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()