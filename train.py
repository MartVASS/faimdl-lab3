import torch
from torch.utils.data import DataLoader
from utils.util_functions import *
from models.custom_model import CustomNet
from tqdm import tqdm
import argparse
import os
import wandb

def main():
    # -------------------------------
    # 1️⃣ Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--dataset_fraction', type=float, default=0.1, help='Fraction of dataset to use')
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.pth', help='Path to save best model')
    args = parser.parse_args()

    # ✅ Définir la clé API directement dans l'environnement
    os.environ["WANDB_API_KEY"] = "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"
    wandb.login()         # variable d'environnement

    # 1️⃣ Initialisation
    wandb.init(
        project="faimdl-lab3",  # nom de ton projet
        config={
            "batch_size": args.batch_size,
            "learning_rate": 1e-3,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "dataset_fraction": args.dataset_fraction
        }
    )
    config = wandb.config
    # -------------------------------
    # 2️⃣ Setup device
    # -------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # -------------------------------
    # 3️⃣ Load & prepare data
    # -------------------------------
    check_and_download_data()
    adjust_data()
    train_dataset, test_dataset = split_dataset()
    
    # Reduce dataset for fast testing
    train_dataset = reduce_dataset(train_dataset, args.dataset_fraction)
    test_dataset = reduce_dataset(test_dataset, args.dataset_fraction)

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------
    # 4️⃣ Initialize model, criterion, optimizer
    # -------------------------------
    model = CustomNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0

    # -------------------------------
    # 5️⃣ Training loop
    # -------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * running_corrects / len(train_loader.dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

        # -------------------------------
        # 6️⃣ Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / len(test_loader.dataset)
        val_acc = 100.0 * val_corrects / len(test_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch
        })

        # -------------------------------
        # 7️⃣ Save best model
        # -------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            wandb.save("args.save_path")
            print(f"Saved best model to {args.save_path}")

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
    
    wandb.finish()
if __name__ == "__main__":
    main()
