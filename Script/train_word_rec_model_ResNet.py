import tensorflow as tf

# Activer la croissance progressive de la mémoire pour les GPU disponibles
try:
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except:
    pass

from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.model_utils import residual_block
from keras.models import Model

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import TrainLogger
from mltu.tensorflow.metrics import CWERMetric
from keras import layers
from datetime import datetime
from mltu.configs import BaseModelConfigs

import os
from tqdm import tqdm

import sys
import io
import json

# Assurer l'encodage UTF-8 pour stdout et stderr
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# Définir la fonction pour créer et entraîner le modèle
def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")

    # Normaliser les images ici au lieu de l'étape de prétraitement
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x8 = residual_block(x7, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model

# Classe pour stocker les configurations du modèle
class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("Models/reconnaisance_de_mots", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20

# Définir les chemins de projet et de dataset
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_dir, "ProcessedImages")

dataset, vocab, maxlen = [], set(), 0

# Lire les fichiers d'annotations
chiffres = open(os.path.join(project_dir, "chiffres.txt"), "r").readlines()

# Traiter les lignes des annotations
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

# Créer un objet ModelConfigs pour stocker les configurations du modèle
configs = ModelConfigs()

# Enregistrer le vocabulaire et la longueur maximale du texte dans les configurations
configs.vocab = "".join(vocab)
configs.max_text_length = maxlen
configs.save()

# Créer un fournisseur de données pour le dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)

# Diviser le dataset en ensembles d'entraînement et de validation
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augmenter les données d'entraînement avec des transformations aléatoires
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10),
]

# Créer l'architecture du modèle TensorFlow
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

# Compiler le modèle et imprimer le résumé
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

# Définir les callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto")

def safe_print(data):
    try:
        print(data)
    except UnicodeEncodeError as e:
        error_message = f"UnicodeEncodeError: {e} in data: {repr(data)}"
        print(error_message, file=sys.stderr)

# Exemple de mise en place de l'entraînement du modèle avec journalisation détaillée
try:
    history = model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback]
    )
except UnicodeEncodeError as e:
    # Gérer l'exception et imprimer les détails
    error_message = f"UnicodeEncodeError: {e}"
    print(error_message, file=sys.stderr)

# Enregistrer l'historique de l'entraînement dans un fichier JSON
history_dict = history.history
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)

# Sauvegarder le modèle entraîné
model.save("reconnaissance_de_mots_avec_ResNets.h5")

# Enregistrer les ensembles d'entraînement et de validation en fichiers CSV
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
