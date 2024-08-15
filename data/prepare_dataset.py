import torchvision
from pathlib import Path
from tqdm import tqdm
import os
import csv

# Get path of project root
project_root = Path(os.getcwd())

# Load MNIST dataset using PyTorch
raw_path = project_root.joinpath("data")
train = torchvision.datasets.MNIST(root=raw_path, download=True, train=True)
test = torchvision.datasets.MNIST(root=raw_path, download=True, train=False)

# Create folder structure
train_path = raw_path.joinpath("MNIST","preprocessed","train")
test_path = raw_path.joinpath("MNIST","preprocessed","test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Save images to correct folder structure
for i, (img, label) in enumerate(tqdm(train)):
    img_path = train_path.joinpath("img_{:05d}.png".format(i))
    img.save(img_path)

for i, (img, label) in enumerate(tqdm(test)):
    img_path = test_path.joinpath("img_{:05d}.png".format(i))
    img.save(img_path)

# Create CSV files with labels
train_csv = train_path.joinpath("labels.csv")
test_csv = test_path.joinpath("labels.csv")
fields = ['img_path', 'label']

print(train_path)

train_labels = [{'img_path': str(train_path.joinpath("img_{:05d}.png".format(i))), 'label': label} for i, (img, label) in enumerate(train)]

with open(train_csv, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(train_labels)

test_labels = [{'img_path': str(test_path.joinpath("img_{:05d}.png".format(i))), 'label': label} for i, (img, label) in enumerate(test)]

with open(test_csv, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields)

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(test_labels)

print("Dataset prepared!")

