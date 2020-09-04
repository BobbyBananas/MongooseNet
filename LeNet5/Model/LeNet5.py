import torch
import torch.nn as nn
import torch.nn.functional as F

"""---------------------------------------------
** LeNet5 Network Model:
---------------------------------------------"""


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Hidden Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # Pass through one layer with 6 filters and a 5x5 kernel, stride is 1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully connected layers
        # Linear function ads the weights[x] and biases[y] y = xA^T + by
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16 channels x 5*5 Reduced Dimensionality
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Define the composition of operations:

        # First Convolution Layer [Three operations]
        x = self.conv1(x)  # Convolution Operation
        x = F.relu(x)  # ReLU Activation Operation
        x = F.max_pool2d(x, 2, 2)  # Pooling Operation (a 2x2 kernel and a stride of 2)

        # Second Convolution Layer
        x = self.conv2(x)  # Convolution Operation
        x = F.relu(x)  # ReLU Activation Operation
        x = F.max_pool2d(x, 2, 2)  # Pooling Operation (a 2x2 kernel and a stride of 2)

        #  Reshape Layer  -> Always reshape to bridge convolutional and fully connected layers
        #  [16 Channels from conv 2,
        x = x.reshape(-1, 16 * 5 * 5)  # the size -1 is inferred from other dimensions

        # First Fully Connected Layer
        x = self.fc1(x)
        x = F.relu(x)

        # Second Fully Connected Layer
        x = self.fc2(x)
        x = F.relu(x)

        # Output layer
        x = self.fc3(x)

        return x
