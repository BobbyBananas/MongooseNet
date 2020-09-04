import torch
import torch.nn as nn
import torch.nn.functional as F

"""------------------------------------------------------------------------------------------
**  Inception Network Model:
    Implementation of a InceptionA Block in the Inception.V3 Model
    Convolutions are performed in parallel and concatenated -> saves computer performance
------------------------------------------------------------------------------------------"""


class InceptionNet(nn.Module):

    def __init__(self):
        super(InceptionNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)         # Pass through one layer  with 10 filters and a 5x5 kernel
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)        # Pass through ten layers with 20 filters and a 5x5 kernel

        self.incept1 = InceptionBlock(in_channels=10, hidden_channels=16, out_channels=24)        # Pass through ten layers into the inception layer
        self.incept2 = InceptionBlock(in_channels=20, hidden_channels=16, out_channels=24)        # Pass through twenty layers into the inception layer

        self.mp = nn.MaxPool2d(2)                            # Pooling Layer of [2x2] kernel
                                                             # Linear function ads the weights[x] and biases[y]      z = xA^T + by
        self.fc = nn.Linear(1408, 10)                        # Nine output nodes with 1408 <=

    def forward(self, x):
        # Save the batch size for reshaping
        in_size = x.size(0)

        # 1 Pass through one layer with 10 filters and a 5x5 kernel
        x = self.conv1(x)
        x = F.relu(self.mp(x))

        # Inception Layer
        x = self.incept1(x)

        # 3
        x = self.conv2(x)
        x = F.relu(self.mp(x))

        # Inception Layer
        x = self.incept2(x)

        #  Reshape Layer
        x = x.view(in_size, -1)  # flatten the tensor

        # Output Fully Connected Layer
        x = self.fc(x)

        return x


"""-------- Inception Block Model----------------

                => [1X1] CONV       
                => [1X1] CONV        => [5X5] CONV                                   
PREVIOUS LAYER                                     => CONCATENATE
                => [1X1] CONV        => [3X3] CONV
                => [3x3] POOL        => [1x1] CONV
                           
---------------------------------------------"""


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels): # Out = 24, Hidden = 16
        super(InceptionBlock, self).__init__()
        # The [1x1] Convolutional layer
        self.branch1x1_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # The [3x3] Convolutional layer
        self.branch3x3_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

        # The [5x5] Convolutional layer
        self.branch5x5_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, out_channels, kernel_size=5, padding=2)

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
