from __future__ import print_function
import argparse
import torch
import numpy as np

# Import dataset and algorithm builder classes
from Evaluation import eval_all
from Builder import Builder

# Import Database UI
from MNIST.Database import Dataset
from MNIST.Database import initialise_data
# ------------------------------------------------------------------------------------------------------------------------
# USER INTERFACE:

# DATASET SIZE
train_size = 60000
validation_size = 10000
test_size = 10000

# TRAINING SETTINGS
total_epoch = 10

# OPTIMIZER SETTINGS
learning_rate = 0.01
momentum = 0.9

# BATCH SIZE
train_batch = 200
valid_batch = train_batch
test_batch = train_batch

# ------------------------------------------------------------------------------------------------------------------------
# Print Options
np.set_printoptions(linewidth=10000, precision=4)
np.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=10000, precision=2, sci_mode=False)
torch.set_printoptions(threshold=10000)
# ------------------------------------------------------------------------------------------------------------------------

# Load the Data
initialise_data(train_size, test_size)

# Train, Test and Evaluate LeNet Algorithm
Algorithm = 'LeNet5'
classifier = Builder(Algorithm, total_epoch, train_batch, learning_rate, momentum)
classifier.runthrough()

# Load Our Mongoose Model
Algorithm = 'Mongoose'
classifier = Builder(Algorithm, total_epoch, train_batch, learning_rate, momentum)
classifier.load_model()
classifier.evaluate()
classifier.confusion()

# Evaluate Models Algorithms for the first epoch
eval_all(True, 0)

# Evaluate Models Algorithms for all epochs
eval_all(False)
