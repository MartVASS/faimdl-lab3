import os
import shutil
import zipfile
import numpy as np
import urllib.request


# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

# Function to load the dataset TinyImage200

def check_and_download_data():
    dataset_path = "dataset/tiny-imagenet-200"
    zip_path = "dataset/tiny-imagenet-200.zip"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading in progress...")
        os.makedirs("dataset", exist_ok=True)
        
        print(f"📡 Downloading from {url}...")
        # Cette ligne télécharge le fichier physiquement
        urllib.request.urlretrieve(url, zip_path)

        print("📦 Extracting files (this operation can take 2-3 minutes)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        print("🧹 Cleaning ZIP file...")
        os.remove(zip_path)
        print("✅ Finish, you can find the dataset at:", dataset_path)
    else:
        
        # On vérifie si le dossier ET un fichier clé existent
        if os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, "words.txt")):
            print("✅ Dataset already extracted and complete.")


def adjust_data():


    with open('dataset/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt') as f:
        
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'dataset/tiny-imagenet-200/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'dataset/tiny-imagenet-200/tiny-imagenet-200/val/images/{fn}', f'dataset/tiny-imagenet-200/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('dataset/tiny-imagenet-200/tiny-imagenet-200/val/images')
    print("succesfully adjusting data")