from utils.util_functions import *

# 1. Get data ready

check_and_download_data()
adjust_data()  # To allow using Image Folder on data

# 2. Split data into training and test sets

train_dataset, test_dataset = split_dataset()

print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of test dataset: {len(test_dataset)}")