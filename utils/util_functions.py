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
    base_path = 'dataset/tiny-imagenet-200/tiny-imagenet-200/val'
    source_dir = os.path.join(base_path, 'images')
    annot_file = os.path.join(base_path, 'val_annotations.txt')

    # On vérifie si le dossier 'images' existe. 
    # S'il n'existe plus, c'est que le script a déjà tourné avec succès.
    if not os.path.exists(source_dir):
        print("Data already adjusted. Skipping.")
        return

    with open(annot_file) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            dest_folder = os.path.join(base_path, cls)
            
            os.makedirs(dest_folder, exist_ok=True)
            
            # On définit les chemins source et destination
            src_path = os.path.join(source_dir, fn)
            dst_path = os.path.join(dest_folder, fn)
            
            # On déplace le fichier au lieu de le copier (plus rapide et évite les doublons)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

    # Une fois que tous les fichiers sont déplacés, on supprime le dossier vide
    shutil.rmtree(source_dir)
    print("succesfully adjusting data")

    