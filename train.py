import torch

from utils.util_functions import *

def main():
    # 1. Get data ready

    check_and_download_data()
    adjust_data()  # To allow using Image Folder on data

    # 2. Split data into training and test sets

    train_dataset, test_dataset = split_dataset()

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4, persistent_workers = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 4, persistent_workers = True)

    # Check out what's inside the training dataloader
    train_features_batch, train_labels_batch = next(iter(train_loader))
    print(train_features_batch.shape) 
    print(train_labels_batch.shape)

if __name__ == '__main__':
    main()