import torch

from utils.util_functions import *
from models.custom_model import CustomNet

def main():
    # 1. Get data ready

    check_and_download_data()
    adjust_data()  # To allow using Image Folder on data

    # 2. Split data into training and test sets

    train_dataset, test_dataset = split_dataset()

    class_names = train_dataset.classes

    train_dataset = reduce_dataset(train_dataset, 0.1)
    test_dataset = reduce_dataset(test_dataset, 0.1)

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers = 2, pin_memory = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers = 2, pin_memory = False)

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(train_loader))
    print(train_features_batch.shape) 
    print(train_labels_batch.shape)

    view_image(train_features_batch, train_labels_batch, class_names) # Viewing a random image

    # 3. Import model 

    # Set up device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    model_lab = CustomNet().to(device) 

    # Create a dummy tensor to test the model 
    dt = torch.randn(size=(3,224,224))
    y_pred = model_lab(dt.unsqueeze(0).to(device))
    print(y_pred.shape)

    # 4. Train and test the model

    model = model_lab.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0

    # Run the training process for {num_epochs} epochs
    num_epochs = 3
    for epoch in tqdm(range(1, num_epochs + 1)):
        train(epoch, model, train_loader, criterion, optimizer, device)

        # At the end of each training iteration, perform a validation step
        val_accuracy = validate(model, test_loader, criterion, device)

        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)


    print(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()