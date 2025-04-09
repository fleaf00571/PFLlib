import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

# Set random seeds for reproducibility
random.seed(1)
np.random.seed(1)

# Default number of clients
num_clients = 100
# Directory path for saving CIFAR-100 (superclass) dataset
dir_path = "Cifar100/"

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory paths for configuration and train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    
    # If dataset with given parameters already exists, skip generation
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
    
    # Load CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=os.path.join(dir_path, "rawdata"),
        train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=os.path.join(dir_path, "rawdata"),
        train=False, download=True, transform=transform
    )
    # Load all data into memory
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data
    
    # Combine train and test data
    dataset_image = []
    dataset_label = []
    # Convert torch tensors to numpy and extend lists
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    
    # Map fine labels (0-99) to coarse labels (0-19) using official CIFAR-100 superclass mapping
    fine_to_coarse = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                      3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                      6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                      0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                      5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                      16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                      10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                      2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                      16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                      18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
    dataset_label = [fine_to_coarse[int(lbl)] for lbl in dataset_label]
    
    # Convert lists to numpy arrays
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    
    # Determine number of classes (should be 20 for coarse labels)
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    
    # Separate data among clients (IID or non-IID based on parameters)
    X, y, statistic = separate_data((dataset_image, dataset_label),
                                    num_clients, num_classes,
                                    niid, balance, partition,
                                    class_per_client=2)
    # Split data into train/test sets for each client
    train_data, test_data = split_data(X, y)
    # Save data and configuration to files
    save_file(config_path, train_path, test_path,
              train_data, test_data,
              num_clients, num_classes,
              statistic, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    
    generate_dataset(dir_path, num_clients, niid, balance, partition)
