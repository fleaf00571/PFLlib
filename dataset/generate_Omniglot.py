import numpy as np
import os
import random
import torchvision
from utils.dataset_utils import split_data, save_file
from PIL import Image


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Omniglot/"


# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = dir_path+"rawdata"
    
    # Get Omniglot data
    torchvision.datasets.Omniglot(root=root, background=True, download=True)
    torchvision.datasets.Omniglot(root=root, background=False, download=True)

    X = [[] for _ in range(20)]
    y = [[] for _ in range(20)]

    dir = os.path.join(root, "omniglot-py/")
    dirs = os.listdir(dir)
    label = 0
    for ddir in dirs:
        if '.' not in ddir:
            ddir = os.path.join(dir, ddir)
            ddirs = os.listdir(ddir)
            for dddir in ddirs:
                if '.' not in dddir:
                    dddir = os.path.join(ddir, dddir)
                    dddirs = os.listdir(dddir)
                    for ddddir in dddirs:
                        ddddir = os.path.join(dddir, ddddir)
                        file_names = os.listdir(ddddir)
                        for i, fn in enumerate(file_names):
                            fn = os.path.join(ddddir, fn)
                            img = Image.open(fn)
                            X[i].append(np.expand_dims(np.asarray(img), axis=0))
                            y[i].append(label)
                    label += 1
                    
    print(f'Number of labels: {label}')
    
    statistic = []
    for i in range(num_clients):
        statistic.append([])
        y_arr = np.array(y[i])
        for yc in sorted(np.unique(y_arr)):
            statistic[-1].append((int(yc), int(sum(y_arr == yc))))

    for i in range(num_clients):
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, label, 
        None, None, None, None)


if __name__ == "__main__":
    generate_dataset(dir_path)