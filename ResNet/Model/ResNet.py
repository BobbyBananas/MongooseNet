from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

"""---------------------------------------------
** Residual Network Model:
---------------------------------------------"""


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        # Residual Layer parameters
        self.layer1 = ResNetLayer(1, 8, 16, 3, padding=1, stride=1)
        self.layer2 = ResNetLayer(16, 32, 16, 5, padding=16, stride=2)

        # Fully connected layer parameters
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Save the batch size for reshaping
        in_size = x.size(0)

        # Pass tensor through the two residual layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Average Pooling with a 4x4 kernel.
        x = F.avg_pool2d(x, kernel_size=4)   # Always performed after residual layers

        #  Reshape Layer -> Always reshape to bridge convolutional and fully connected layers
        x = x.view(in_size, -1)

        # First Fully Connected Layer
        x = self.fc1(x)
        x = F.relu(x)

        # Second Fully Connected Layer
        x = self.fc2(x)

        # Output layer
        return x


"""---------------------------------------------
** Basic Residual Block:
---------------------------------------------"""


class ResNetLayer(nn.Module):

    def __init__(self, in_channel: int, hidden_channel: int,  out_channel: int, kernel: int, padding: int, stride: int):
        super(ResNetLayer, self).__init__()

        # Define the convolution layer parameters
        self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel, stride, padding)
        self.conv_2 = nn.Conv2d(hidden_channel, out_channel, kernel, stride, padding)

        # Define the batch norm layers
        self.batchnorm_1 = nn.BatchNorm2d(hidden_channel)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        # Make a copy on which to perform convolutions
        y = x

        # Perform Convolution -> BatchNorm
        y = self.conv_1(y)
        y = self.batchnorm_1(y)
        y = F.relu(y)  # Activation

        # Perform Convolution -> BatchNorm
        y = self.conv_2(y)
        y = self.batchnorm_2(y)
        y = F.relu(y)  # Activation

        return x + y  # skip connection

