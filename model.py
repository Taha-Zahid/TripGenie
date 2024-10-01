import torch  # Importing the PyTorch library for building and training neural networks
import torch.nn as nn  # Importing the neural network module from PyTorch

class NeuralNet(nn.Module):
    # Implementing a feed-forward neural network
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Initializing the base class (nn.Module)
        
        # Defining the layers of the neural network
        self.l1 = nn.Linear(input_size, hidden_size)  # First layer: input to hidden
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second layer: hidden to hidden
        self.l3 = nn.Linear(hidden_size, num_classes)  # Third layer: hidden to output (num_classes)

        self.relu = nn.ReLU()  # Defining the activation function (ReLU)

    def forward(self, x):
        # Defining the forward pass of the network
        out = self.l1(x)  # Pass input through the first layer
        out = self.relu(out)  # Apply ReLU activation function
        out = self.l2(out)  # Pass through the second layer
        out = self.relu(out)  # Apply ReLU activation function
        out = self.l3(out)  # Pass through the final layer
        # No activation or softmax needed here, as the output will be processed by a loss function later
        return out  # Returning the final output logits
