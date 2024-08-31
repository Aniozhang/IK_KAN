from kan import *
import torch
import numpy as np
import os 
from torch.utils.data import TensorDataset, random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

datasetPath = "./dataset"

def create_torch_dataset(X, Y, ratio):
    """
    Convert NumPy arrays X and Y into PyTorch datasets and split them into training and testing sets.

    Parameters:
        X (np.ndarray): Input features array of shape (n_samples, n_features).
        Y (np.ndarray): Output labels array of shape (n_samples, n_labels).
        ratio (float): Ratio of the dataset to be used as the training set.

    Returns:
        dict: A dictionary with keys 'train_input', 'train_output', 'test_input', 'test_output',
              each containing the corresponding PyTorch dataset.
    """
    # Convert the NumPy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Create a full dataset
    dataset = TensorDataset(X_tensor, Y_tensor)

    # Calculate the number of training samples
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Extract input and output datasets from train and test splits
    train_input, train_output = zip(*train_dataset)
    test_input, test_output = zip(*test_dataset)

    # Convert back to tensor datasets
    train_input = torch.stack(train_input).to(device)
    train_output = torch.stack(train_output).to(device)
    test_input = torch.stack(test_input).to(device)
    test_output = torch.stack(test_output).to(device)

    return {
        "train_input": train_input,
        "train_label": train_output,
        "test_input": test_input,
        "test_label": test_output
    }
    

def trainForRobot(dof):
    ## Load the dataset
    dataDir = str(dof) + "_robot"
    dataDirPath = os.path.join(datasetPath, dataDir)

    inputFilePath = dataDirPath + "/datasetInput.npy"
    outputFilePath = dataDirPath + "/datasetOutput.npy"

    X = np.load(inputFilePath)
    Y = np.load(outputFilePath)

    dataset = create_torch_dataset(X, Y, 0.8)

    ## Folder to save at
    os.makedirs(os.path.dirname("./kanModels/"), exist_ok=True)
    os.makedirs(os.path.dirname("./kanModels/kan_" + str(dof)), exist_ok=True)
    modelFolder = "./kanModels/kan_" + str(dof)

    ## The model's input will always be the 6 parameters required to discribe the end effector
    ## The model's output will be the same number as the degree of freedoms. 
    model = KAN(width=[dof,dof**2,6], grid=3, k=3, seed=0, device=device)

    model.fit(dataset, steps=20) # ,img_folder=modelFolder
    model.prune()
    model.plot()

trainForRobot(4)

if __name__ == "__main__":
    for dof in range(4, 50):
        if device == "cuda":
            try:
                trainForRobot(dof)
            except torch.cuda.OutOfMemoryError:
                device = "cpu"
                trainForRobot(dof)
        trainForRobot(dof)
        