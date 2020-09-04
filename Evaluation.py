from __future__ import print_function
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Gather Algorithms directory info
train_indx_directory = 'Results/train_loss_indx.npy'


def eval_all(loss, epoch=0 , accuracy=True):

    # Load All Results
    lenet_vals = np.load('LeNet5/Results/train_loss_epochs.npy')
    incept_vals = np.load('Inception/Results/train_loss_epochs.npy')
    resnet_vals = np.load('ResNet/Results/train_loss_epochs.npy')
    mongoose_vals = np.load('Mongoose/Results/train_loss_epochs.npy')

    loss_index = np.load(train_indx_directory)
    fig = plt.figure()

    if loss:  # Show data over one epoch
        loss_index = loss_index[epoch]
        lenet_vals = lenet_vals[epoch]
        incept_vals = incept_vals[epoch]
        resnet_vals = resnet_vals[epoch]
        mongoose_vals = mongoose_vals[epoch]

    else:  # Show data over all epochs

        # Concatenate all epoch data
        loss_index = loss_index.flatten()
        lenet_vals = lenet_vals.flatten()
        incept_vals = incept_vals.flatten()
        resnet_vals = resnet_vals.flatten()
        mongoose_vals = mongoose_vals.flatten()

    # Plot all loss data
    plt.plot(loss_index, lenet_vals, color='blue')
    plt.plot(loss_index, incept_vals, color='red')
    plt.plot(loss_index, resnet_vals, color='green')
    plt.plot(loss_index, mongoose_vals, color='yellow')

    plt.legend(['LeNet5', 'InceptionNet', 'ResNet', 'Mongoose'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('loss')
    plt.title('Loss Vs Iterations')

    fig.show()
    fig.savefig('Results/train_loss.png')

    if accuracy:  # Show data over all epoch
        # Accuracy Graph
        lenet_vals = np.load('LeNet5/Results/correct_epochs.npy')
        incept_vals = np.load('Inception/Results/correct_epochs.npy')
        resnet_vals = np.load('ResNet/Results/correct_epochs.npy')
        mongoose_vals = np.load('Mongoose/Results/correct_epochs.npy')

        fig = plt.figure()

        lenet_vals = lenet_vals[0]
        incept_vals = incept_vals[0]
        resnet_vals = resnet_vals[0]
        mongoose_vals = mongoose_vals[0]

        # Plot all loss data
        plt.plot(lenet_vals, color='blue')
        plt.plot(incept_vals, color='red')
        plt.plot(resnet_vals, color='green')
        plt.plot(mongoose_vals, color='yellow')

        plt.legend(['LeNet5', 'InceptionNet', 'ResNet', 'Mongoose'], loc='lower right')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')

        fig.show()
        fig.savefig('Results/Accuracy_epoch.png')