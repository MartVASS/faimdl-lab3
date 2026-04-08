import os
import zipfile
import numpy as np
import urllib


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
        print("Dataset non trouvé. Téléchargement en cours...")
        os.makedirs("dataset", exist_ok=True)
        
        # Téléchargement
        urllib.request.urlretrieve(url, zip_path)
        
        # Extraction
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset/")
        
        # Nettoyage : on supprime le ZIP pour gagner de la place
        os.remove(zip_path)
        print("Extracting data completed, ZIP file deleted.")
    else:
        
        print("Dataset found ready to extract data...")
        # Extraction
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset/")
        
        # Nettoyage : on supprime le ZIP pour gagner de la place
        os.remove(zip_path)
        print("Extracting data completed, ZIP file deleted")

    print("Datas ready !")