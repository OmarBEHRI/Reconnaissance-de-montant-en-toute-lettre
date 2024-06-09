import os
from tqdm import tqdm

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