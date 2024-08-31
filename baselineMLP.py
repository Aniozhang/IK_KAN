from kan import *
import torch
import numpy as np
import os, json
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

datasetPath = "./dataset"
num_epochs = 50

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

    train_dataset = TensorDataset(train_input, train_output)
    test_dataset = TensorDataset(test_input, test_output)

    return train_dataset, test_dataset


class FlexibleMLP(nn.Module):
    def __init__(self, layer_sizes, activations):
        super(FlexibleMLP, self).__init__()
        
        # Check if the number of layers and activations match
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("The number of activation functions must be one less than the number of layer sizes.")
        
        # Initialize the layers and activations
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            activation = activations[i]
            if isinstance(activation, str):
                activation = getattr(nn, activation)()  # Converts string name to an actual nn.Module
            layers.append(activation)
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def trainMLP(dof, num_epochs=50):
    # Example usage:
    layer_sizes = [dof, dof**2, 6]  # Input layer, two hidden layers, output layer
    activations = [nn.ReLU(), nn.ReLU()]  # ReLU for hidden layers, no activation for output

    model = FlexibleMLP(layer_sizes, activations).to(device)

    ## Load the dataset
    dataDir = str(dof) + "_robot"
    dataDirPath = os.path.join(datasetPath, dataDir)

    inputFilePath = os.path.join(dataDirPath, "datasetInput.npy")
    outputFilePath = os.path.join(dataDirPath, "datasetOutput.npy")

    X = np.load(inputFilePath)
    Y = np.load(outputFilePath)

    train_dataset, test_dataset = create_torch_dataset(X, Y, 0.8)

    # Set your batch size
    batch_size = 64  # You can adjust this according to your requirements

    # Create DataLoader for training and testing data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        # Print the losses for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    model_save_path = f"mlpModels/mlp_{dof}.model"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot and save the training and validation loss curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the plot as a .png file
    plot_save_path = f"mlpModels/loss_plot_{dof}.png"
    plt.savefig(plot_save_path)
    print(f"Loss plot saved to {plot_save_path}")

    #plt.show()

    # Save the loss data to a JSON file
    loss_data = {
        "Name": f"MLP_{dof}",
        "TrainLoss": train_losses,
        "ValLoss": val_losses,
        "FinalTrainLoss": train_losses[-1],
        "FinalValLoss": val_losses[-1]
    }

    json_save_path = f"mlpModels/loss_data_{dof}.json"
    if os.path.exists(json_save_path):
        with open(json_save_path, 'r+') as file:
            data = json.load(file)
            if isinstance(data, list):
                data.append(loss_data)
            else:
                data = [data, loss_data]
            file.seek(0)
            json.dump(data, file, indent=4)
    else:
        with open(json_save_path, 'w') as file:
            json.dump([loss_data], file, indent=4)
    
    print(f"Loss data saved to {json_save_path}")

for dof in range(4, 50):
    trainMLP(dof)