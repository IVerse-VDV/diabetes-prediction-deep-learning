"""
model.py

This file defines the architecture of a simple feedforward neural network using PyTorch.
The model is designed to perform binary classification to predict whether a patient has diabetes or not,
based on 8 medical input features.

Architecture:
- Input Layer: 8 features
- Hidden Layer 1: 16 units, ReLU activation
- Hidden Layer 2: 8 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation (for binary classification)
"""




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np





class DiabetesModel(nn.Module):
    """
    A simple feedforward neural network for predicting diabetes diagnosis
    using 8 input features.

    Architecture:
    - Input layer (size = 8)
    - Hidden layer 1: Linear(8 => 16) + ReLU
    - Hidden layer 2: Linear(16 => 8) + ReLU
    - Output layer: Linear(8 => 1) + Sigmoid (for binary classification)
    """

    def __init__(self, input_size=8):
        super(DiabetesModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 16),  # Maps input features to 16 hidden units
            nn.ReLU(),                  # Applies non linearity
            nn.Linear(16, 8),           # Reduces to 8 units
            nn.ReLU(),                  # Another non linearity
            nn.Linear(8, 1),            # Final output unit (binary classification)
            nn.Sigmoid()                # Squashes output to [0, 1] for probability
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Output probabilities between 0 and 1
        """
        return self.model(x)
    




class DiabetesTrainer:
    """
    Trainer class for handling the training, evaluation, prediction,
    saving, and loading of a pytorch model for diabetes classification
    """




    def __init__(self, model, learning_rate=0.001):
        # Init the trainer with the given model, loss function, optimizer, and training log
        self.model = model
        self.criterion = nn.BCELoss()  # bce loss for binary classification
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
        self.train_losses = []  #to store loss values per epoch




    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Trains the model on the provided training data.

        This method performs the full training loop over the dataset for a given number
        of epochs. It uses mini batch gradient descwnt with a specified batch size, and
        updates model weights using backpropagation and the adam optimizer.

        AArgs:
            X_train (np.ndarray): The input features for training. Each row represents a sample, 
                                  and each column represents a feature (glucose level, age, etc)
            y_train (np.ndarray): The ground truth labels (0 or 1) indicating whether the patient
                                  is diabetic (1) or not (0).
            epochs (int): The number of times the model will iterate over the entire training dataset
            batch_size (int): The number of samples the model processes at once before updating the weights

        Example:
            trainer.train_model(X_train, y_train, epochs=100, batch_size=64)
        """

        # Convert training data from numpy arrays to pytorch tensors
        # This is necessary because the model and training operations use PyTorch data types
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))  # Reshape labels to (N, 1) for bceLoss compatibility

        # Combine features and labels into a pytorch dataset, which allows efficient batching
        dataset = TensorDataset(X_tensor, y_tensor)

        # Use a DataLoader to handle mini batching and shuffling of the dataset for each epoch
        # Shuffling helps prevent the model from learning spurious patterns in the order of the data
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Set the model to training mode
        # This enables features like dropout (if used) and gradient tracking
        self.model.train()

        # Training loop over all epochs
        for epoch in range(epochs):
            epoch_loss = 0.0  # Accumulate the loss for each batch to track progress

            # Loop over the dataset in batches
            for batch_X, batch_y in dataloader:
                # === Forward pass ===
                # Feed the batch input through the model to get predicted probabilities
                outputs = self.model(batch_X)

                # Compute the loss between predicted outputs and actual labels
                # Here, we use Binary Cross Entropy because this is a binary classification task
                loss = self.criterion(outputs, batch_y)

                # === Backward pass & optimization ===
                self.optimizer.zero_grad()  # Reset gradients from previous step
                loss.backward()             # Compute gradients via backpropagation
                self.optimizer.step()       # Uodate model parameters using gradients

                # Accumulate the loss value for reporting
                epoch_loss += loss.item()

            # Calculate the average loss over all batches in this epoch
            avg_loss = epoch_loss / len(dataloader)

            # record the average loss so it can be plotted or saved later
            self.train_losses.append(avg_loss)

            # Optionally print training progress every 10 epochs to track performance
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')




    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained model on a given test dataset and returns the accuracy.
    
        This method is typically used after training to assess how well the model 
        generalizes to unseen data. It performs a forward pass through the model 
        without computing gradients (to save memory and computation), applies a 
        classification threshold of 0,5 to the output probabilities, and compares 
        the predicted labels with the ground truth to calculate accuracy.
    
        Args:
            X_test (np.ndarray): A NumPy array of shape (n_samples, n_features) 
                containing the feature values of the test dataset.
            y_test (np.ndarray): A NumPy array of shape (n_samples,) or 
                (n_samples, 1) containing the true labels (0 or 1) for each test sample.
    
        Returns:
            float: A scalar representing the classification accuracy of the model 
            on the test dataset, expressed as a value between 0.0 (0%) and 1.0 (100%).
        """
        # Set the model to evaluation mode
        # This disables dropout and batch normalization layers if present,
        # ensuring consistent output during inference
        self.model.eval()
    
        # Disable gradient calculation for all operations within this block
        # This is important during evaluation to reduce memory usage and speed up computation
        # as we dont need gradients for backpropagation
        with torch.no_grad():
            # Convert the NumPy test data into PyTorch tensors.
            # Ensure that the target labels (y_test) are reshaped to a column vector (n_samples, 1),
            # which matches the output shape of the model.
            X_tensor = torch.FloatTensor(X_test)
            y_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
            # Run a forward pass through the model to get output probabilities.
            outputs = self.model(X_tensor)
    
            # Apply a threshold of 0.5 to convert probabilities into binary class predictions (0 or 1).
            # Values above 0.5 are classified as class 1 (positive for diabetes),
            # and those below or equal to 0.5 are classified as class 0 (negative).
            predictions = (outputs > 0.5).float()
    
            # Compare the predicted labels with the true labels (element-wise equality),
            # convert boolean results to float (True → 1.0, False → 0.0),
            # and compute the mean to get the accuracy.
            accuracy = (predictions == y_tensor).float().mean()
    
        # Return the accuracy as a regular Python float value.
        return accuracy.item()




    def predict_probability(self, X):
        """
        Predicts the probability of a positif diabetes diagnosis for new input data

        This function can handle either a single inpyt sample or a batch of samples
        It processes the input, ensures the data is in the correct shape, and performs 
        a forward pass through the model to generate a probability score between 0 and 1.

        It does not apply any thresholding or classification logic,  it simply returns 
        the raw predicted probability output from the sigmoid activation function 
        in the model.

        Args:
            X (np.ndarray or list): New input data, either a single sample 
                (1D array or list of shape [n_features]) or multiple samples 
                (2D array of shape [n_samples, n_features]).

        Returns:
            float or np.ndarray: The predicted probability/probabilities. 
            Returns a single float if input is one sample, or a NumPy array 
            of floats if input is a batch.
        """
        # set the model to evaluation mode to disable dropout, etc
        self.model.eval()

        # Disable gradient computation for inference (memory efficient)
        with torch.no_grad():
            # Handle both single sample and batch of samples
            if isinstance(X, np.ndarray):
                # If input is a 1D array (single sample), reshape to 2D for model input
                # If already 2D (batch), use it directly
                X_tensor = torch.FloatTensor(X.reshape(1, -1) if X.ndim == 1 else X)
            else:
                # Assume its a single sample given as a Python list
                # Convert to a 2D tensor of shape (1, n_features)
                X_tensor = torch.FloatTensor([X])

            # Pass the input tensor through the model to get the predicted probability
            probability = self.model(X_tensor)

            # If it's a single sample, return a scalar float value
            if X_tensor.shape[0] == 1:
                return float(probability.item())
            else:
                # For multiple samples, squeeze unnecessary dimensions
                # and convert to NumPy array of floats
                return probability.squeeze().numpy().astype(float)





    def save_model(self, filepath):
        """
        Saves the model current state to a file, including its learned weights,
        optimizer parameters, and training loss history.

        This function is useful for checkpointing during or after training so that
        the model can be reloaded later without having to retrain from scratch.

        What gets saved:
        - 'model_state_dict': contains the weights and biases of the model
        - 'optimizer_state_dict': contains the state of the optimizer, including
          momentum, learning rate, other internal buffers, etc (Check in 'train_model')
        - 'train_losses': a list of the average loss from each epoch, useful for
          tracking training progress

        Args:
            filepath (str): The full path (including filename) where the model 
            checkpoint will be saved. Example: 'models/diabetes_model.pth'
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),         # Save all model weights
            'optimizer_state_dict': self.optimizer.state_dict(), # Save optimizer internal state
            'train_losses': self.train_losses                    # Save training loss log
        }, filepath)





    def load_model(self, filepath):
        """
        Loads the model, optimizer state, and training loss history from a file about this

        Args:
            filepath (str): Path to the saved model checkpoint
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
