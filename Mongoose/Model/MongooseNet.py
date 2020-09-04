import torch
import torch.nn as nn
import torch.nn.functional as F


"""---------------------------------------------
** Mongoose Network Model:
---------------------------------------------"""


class MongooseNet(nn.Module):

    def __init__(self):
        super(MongooseNet, self).__init__()

        # Residual Layers
        self.layer1 = MongooseLayer(1, 8, 16, 3, padding=1, stride=1)
        self.layer2 = MongooseLayer(16, 32, 16, 5, padding=16, stride=2)

        # Fully Connected Layer
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Save the batch size for reshaping
        in_size = x.size(0)

        # The two residual layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Average Pooling with a 4x4 kernel.
        x = F.avg_pool2d(x, kernel_size=4)
        x = F.elu(x)

        #  Reshape Layer
        x = x.view(in_size, -1)

        # First Fully Connected Layer
        x = self.fc1(x)
        x = F.elu(x)

        # Second Fully Connected Layer
        x = self.fc2(x)
        x = F.elu(x)

        # Output Layer
        x = self.fc3(x)

        return x


"""---------------------------------------------
** Mongoose Residual Block:
---------------------------------------------"""


class MongooseLayer(nn.Module):

    def __init__(self, in_channel: int, hidden_channel: int,  out_channel: int, kernel: int, padding: int, stride: int):
        super(MongooseLayer, self).__init__()

        # Define the Convolutional Layer Parameters
        self.conv_1 = nn.Conv2d(in_channel, hidden_channel, kernel, stride, padding)
        self.conv_2 = nn.Conv2d(88, out_channel, kernel, stride, padding)

        # Define the Inception Layer Parameters
        self.incept1 = InceptionBlock(in_channels=hidden_channel, hidden_channels=16, out_channels=24)

        # Define the batch norm layers
        self.batchnorm_1 = nn.BatchNorm2d(hidden_channel)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        y = x  # Make a copy on input to perform convolutions

        # Perform Convolution -> BatchNorm
        y = self.conv_1(y)
        y = self.batchnorm_1(y) # Improves performance speed by reducing internal covariate shift
        y = F.elu(y)  # Exp Activation

        # Inception Layer
        y = self.incept1(y)

        # Perform Convolution -> BatchNorm
        y = self.conv_2(y)
        y = self.batchnorm_2(y)
        y = F.elu(y)  # Exp Activation

        return x + y  # skip connection


"""--------------------Inception Block Model----------------------
                => [1X1] CONV       
                => [1X1] CONV        => [5X5] CONV                                   
PREVIOUS LAYER                                     => CONCATENATE
                => [1X1] CONV        => [3X3] CONV
                => [3x3] POOL        => [1x1] CONV
---------------------------------------------------------------"""


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # The [1x1] Convolutional layer
        self.branch1x1_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # The [3x3] Convolutional layer
        self.branch3x3_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        # The [5x5] Convolutional layer
        self.branch5x5_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=5, padding=2)

        # The [1x1] Max_Pooling Convolutional layer
        self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Define the composition of operations:

        # The [1x1] Convolutional layer
        branch1x1 = self.branch1x1_1(x)

        # The [3x3] Convolutional layer
        branch3x3dbl = self.branch3x3_1(x)
        branch3x3dbl = self.branch3x3_2(branch3x3dbl)

        # The [5x5] Convolutional layer
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # The [3x3] Max Pooling Layer
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Concatenate the outputs
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(outputs, 1)

        return outputs
