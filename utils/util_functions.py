import os
import shutil
import zipfile
import numpy as np
import urllib.request
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # Import tqdm for progress bar


# Function to denormalize image for visualization
def denormalize(image):
    """Denormalize image for visualization
    
    params: image
    
    return image
    """
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

# Function to load the dataset TinyImage200

def check_and_download_data():
    """ Get dataset from web if not downloaded and extract all files in a new folder"""
    
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

    """Adjust data to be able to use image folder"""
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

def split_dataset():

    """Split the dataset into train set and test set
    
    return train_dataset, test_dataset
    """

    transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet-200/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet-200/tiny-imagenet-200/val', transform=transform)

    return tiny_imagenet_dataset_train, tiny_imagenet_dataset_val

def view_image(train_features_batch, train_labels_batch, class_names):

    """Vizualize one random image given a batch of train and test image and the class names."""
    torch.manual_seed(42)
    random_idx = torch.randint(0, len(train_features_batch), size = [1]).item()
    img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
    plt.imshow(denormalize(img))
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()
    print(f"Image size: {img.shape}")
    print(f"Label: {label}, label size: {label.shape}")


def train(epoch, model, train_loader, criterion, optimizer):
    """Perform a training epoch
    
    params:
    epoch: int - Number of the current epoch
    model: nn.Module - Model to train
    train_loader: torch.utils.data.DataLoader - Training dataloader
    criterion: torch.nn - Loss function
    optimizer: torch.optim - Optimizer
    
    return: None
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):

      inputs, targets = inputs.to(device), targets.to(device)

      # Compute prediction and loss
      outputs = model(inputs)
      loss = criterion(outputs, targets)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Testing loop
def validate(model, val_loader, criterion):
    """Performs a testing loop
    
    params:
    model: nn.Module - Model to test
    val_loader: torch.utils.data.DataLoader - Testing dataloader
    criterion: torch.nn - Loss function

    return:
    val_accuracy: float - Testing accuracy
    """
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.inference_mode():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Testing Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy