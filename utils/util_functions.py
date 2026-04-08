import os
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
        print("Dataset non trouvé. Téléchargement en cours...")
        os.makedirs("dataset", exist_ok=True)
        
        print(f"📡 Téléchargement depuis {url}...")
        # Cette ligne télécharge le fichier physiquement
        urllib.request.urlretrieve(url, zip_path)

        print("📦 Extraction des fichiers (cela peut prendre 2-3 minutes)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        
        print("🧹 Nettoyage du fichier ZIP...")
        os.remove(zip_path)
        print("✅ Terminé ! Le dataset est prêt dans :", dataset_path)
    else:
        
        # On vérifie si le dossier ET un fichier clé existent
        if os.path.exists(dataset_path) and os.path.exists(os.path.join(dataset_path, "words.txt")):
            print("✅ Dataset déjà extrait et complet.")


