from __future__ import print_function
import argparse
import torch
import torch.nn as nn

import torchvision
import random

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn import metrics

# Import Algorithm Models
from Inception.Model.Inception import InceptionNet
from LeNet5.Model.LeNet5 import LeNet5
from ResNet.Model.ResNet import ResNet
from Mongoose.Model.MongooseNet import MongooseNet

# Import Database UI
from MNIST.Database import Dataset


class Builder:
    def __init__(self, algorithm, epoch, batch_size, learning_rate, momentum):
        # Save the parameters
        self.algorithm = algorithm
        self.epoch = epoch
        self.batch_size = batch_size

        # Create The Network
        if algorithm == 'Inception':
            self.model = InceptionNet()
        elif algorithm == 'Mongoose':
            self.model = MongooseNet()
        elif algorithm == 'ResNet':
            self.model = ResNet()
        elif algorithm == 'LeNet5':
            self.model = LeNet5()

        # Store network in cuda if available
        if torch.cuda.is_available():
            self.model.cuda()

        # Build the optimiser
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

        # Gather Algorithms directory info
        self.optimizer_directory = '{}/Results/optimizer.pth'.format(self.algorithm)
        self.model_directory = '{}/Results/model.pth'.format(self.algorithm)
        self.train_loss_directory = '{}/Results/train_loss_epochs.npy'.format(self.algorithm)
        self.train_indx_directory = 'Results/train_loss_indx.npy'
        self.prediction_probability_directory = '{}/Results/predicted_probs.npy'.format(self.algorithm)
        self.prediction_values_directory = '{}/Results/predicted_values.npy'.format(self.algorithm)
        self.train_correct_directory = '{}/Results/correct_epochs.npy'.format(self.algorithm)

        # Initialise Data Names
        self.t_train_img = 0
        self.t_train_label = 0
        self.t_test_img = 0
        self.t_test_label = 0
        self.t_valid_img = 0
        self.t_valid_label = 0

        self.train_size = 0
        self.valid_size = 0
        self.test_size = 0

    def load_data(self):
        # Load the Saved Numpy Arrays
        n_train_img = np.load("MNIST/train_images.npy")
        n_train_label = np.load("MNIST/train_labels.npy")
        n_test_img = np.load("MNIST/test_images.npy")
        n_test_label = np.load("MNIST/test_labels.npy")
        n_valid_img = np.load("MNIST/test_images.npy")
        n_valid_label = np.load("MNIST/test_labels.npy")

        # Store data info
        self.train_size = len(n_train_label)
        self.valid_size = len(n_valid_label)
        self.test_size = len(n_test_label)

        # Modify Dataset Depending on the Algorithm
        if self.algorithm == 'LeNet5':
            # Input Dimensions are 28x28 add 2 rows on either side to get 32x32 dimension
            n_train_img = np.pad(n_train_img, ((0, 0), (0, 0), (2, 2), (2, 2)), 'edge')
            n_test_img = np.pad(n_test_img, ((0, 0), (0, 0), (2, 2), (2, 2)), 'edge')
            n_valid_img = np.pad(n_valid_img, ((0, 0), (0, 0), (2, 2), (2, 2)), 'edge')

        # Convert Numpy File to Tensor
        t_train_img = torch.from_numpy(n_train_img)
        t_train_label = torch.from_numpy(n_train_label)
        t_test_img = torch.from_numpy(n_test_img)
        t_test_label = torch.from_numpy(n_test_label)
        t_valid_img = torch.from_numpy(n_valid_img)
        t_valid_label = torch.from_numpy(n_valid_label)

        # Move tensors into gpu if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = 'cpu'

        # Reformat data types
        self.t_train_img = t_train_img.to(device, dtype=torch.float)
        self.t_train_label = t_train_label.to(device, dtype=torch.long)
        self.t_test_img = t_test_img.to(device, dtype=torch.float)
        self.t_test_label = t_test_label.to(device, dtype=torch.long)
        self.t_valid_img = t_valid_img.to(device, dtype=torch.float)
        self.t_valid_label = t_valid_label.to(device, dtype=torch.long)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_directory))
        self.optimizer.load_state_dict(torch.load(self.optimizer_directory))

    def summary(self):
        print("Train Image => {} \nTrain Label => {} \nTest Image  => {} \nTest Label  => {}\n".format(
             self.t_train_img.size(), self.t_train_label.size(), self.t_test_img.size(), self.t_test_label.size()))

    def train(self):
        print(" Training {}".format(self.algorithm))

        # Create Batches of training Data
        batch_img = torch.utils.data.DataLoader(self.t_train_img, batch_size=self.batch_size, shuffle=False)
        batch_label = torch.utils.data.DataLoader(self.t_train_label, batch_size=self.batch_size, shuffle=False)

        # Define the number of batches per training
        iterations = int(self.train_size / self.batch_size)


        # Create Array To store loss data for every epoch
        loss_data = np.empty([self.epoch, iterations])

        # Create Index Array
        loss_index = np.empty_like(loss_data)

        # Create Epoch Accuracy Array
        accuracy_epoch = np.empty([2, self.epoch])

        # Repeat Training for each epoch
        for epoch in range(self.epoch):
            total_loss = 0

            b_img = iter(batch_img)  # Getting a batch
            b_label = iter(batch_label)  # Getting a batch

            for i in range(0, iterations):
                image = next(b_img)
                label = next(b_label)


                # Make the predictions
                preds = self.model(image)  # Pass Batch

                loss = F.cross_entropy(preds, label)  # Calculate Loss

                self.optimizer.zero_grad()
                loss.backward()  # Calculate Gradients
                self.optimizer.step()  # Update Weights

                total_loss += loss.item()

                # Store Batch Loss Data
                loss_data[epoch, i] = loss.item()
                loss_index[epoch, i] = ((i + 1) + (epoch * iterations)) * self.batch_size

                if (i * self.batch_size % 1000) == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, i * self.batch_size, self.train_size,
                        100. * i * self.batch_size / self.train_size, loss.item()))

            # Run Validation and store num correct
            accuracy_epoch[0, epoch] = self.validate(epoch) * 100. / self.valid_size
            accuracy_epoch[1, epoch] = epoch + 1

        # Save Loss Data in appropriate file
        np.save(self.train_loss_directory, loss_data)
        np.save(self.train_indx_directory, loss_index)
        np.save(self.train_correct_directory, accuracy_epoch)

        # Save The Models in appropriate files
        torch.save(self.optimizer.state_dict(), self.optimizer_directory)
        torch.save(self.model.state_dict(), self.model_directory)

    def validate(self, epoch):
        # Create Batches of testing Data
        batch_img = torch.utils.data.DataLoader(self.t_valid_img, batch_size=self.batch_size, shuffle=False)
        batch_label = torch.utils.data.DataLoader(self.t_valid_label, batch_size=self.batch_size, shuffle=False)

        b_img = iter(batch_img)  # Getting a batch
        b_label = iter(batch_label)  # Getting a batch

        test_loss = 0
        correct = 0

        # Initialise Prediction Arrays
        pred_prob = np.array([])
        pred_val = np.array([])

        with torch.no_grad():
            for i in range(0, int(self.valid_size / self.batch_size)):
                image = next(b_img)
                label = next(b_label)

                preds = self.model(image)  # [Batch, 10]

                # Total up losses
                test_loss += F.cross_entropy(preds, label).item()

                # Return the value of the max probability and its index
                prediction = preds.data.max(1, keepdim=False)  # Returns a tuple (max_values, indices) indices match with values

                # Save the predicted values in a numpy array
                prediction_vals = prediction[1].cpu().numpy()
                pred_val = np.append(pred_val, prediction_vals)

                # Save the predicted probability in a numpy array
                prediction_probs = prediction[0].cpu().numpy()
                pred_prob = np.append(pred_prob, prediction_probs)

                # Compare predictions with values to find total correct
                correct += prediction[1].eq(label.data.view_as(prediction[1])).cpu().sum()

        print("\n{} TEST EPOCH {} :\nAverage Loss:\t{:.4} \nCorrect:\t{}/{}\nAccuracy:\t{:.0f}%\n".format(
            self.algorithm, epoch + 1, test_loss / (self.valid_size / self.batch_size), correct, self.valid_size, (correct * 100 / self.test_size)
        ))

        return correct

    def test(self):
        print("Testing {}".format(self.algorithm))

        # Create Batches of testing Data
        batch_img = torch.utils.data.DataLoader(self.t_test_img, batch_size=self.batch_size, shuffle=False)
        batch_label = torch.utils.data.DataLoader(self.t_test_label, batch_size=self.batch_size, shuffle=False)

        b_img = iter(batch_img)  # Getting a batch
        b_label = iter(batch_label)  # Getting a batch

        test_loss = 0
        correct = 0

        # Initialise Prediction Arrays
        pred_prob = np.array([])
        pred_val = np.array([])

        with torch.no_grad():
            for i in range(0, int(self.test_size / self.batch_size)):
                image = next(b_img)
                label = next(b_label)

                preds = self.model(image)  # [Batch, 10]

                # Total up losses
                test_loss += F.cross_entropy(preds, label).item()

                # Return the value of the max probability and its index
                prediction = preds.data.max(1, keepdim=False)  # Returns a tuple (max_values, indices) indices match with values

                # Save the predicted values in a numpy array
                prediction_vals = prediction[1].cpu().numpy()
                pred_val = np.append(pred_val, prediction_vals)

                # Save the predicted probability in a numpy array
                prediction_probs = prediction[0].cpu().numpy()
                pred_prob = np.append(pred_prob, prediction_probs)

                # Compare predictions with values to find total correct
                correct += prediction[1].eq(label.data.view_as(prediction[1])).cpu().sum()

        # Save the predictions
        np.save(self.prediction_values_directory, pred_val)
        np.save(self.prediction_probability_directory, pred_prob)

        print("\n{} TEST:\nAverage Loss:\t{:.4} \nCorrect:\t{}/{}\nAccuracy:\t{:.0f}%\n".format(
            self.algorithm, test_loss / (self.test_size / self.batch_size), correct, self.test_size, (correct * 100 / self.test_size)
        ))

    def evaluate(self, name=''):
        accuracy = np.load(self.train_correct_directory)

        fig = plt.figure()
        plt.plot(accuracy[1], accuracy[0], color='blue')

        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy')
        plt.title('{}'.format(self.algorithm))

        fig.show()
        fig_name = '{}/Results/accuracy_epoch{}.png'.format(self.algorithm, name)
        fig.savefig(fig_name)

        vals = np.load(self.train_loss_directory)
        loss_index = np.load(self.train_indx_directory)

        fig = plt.figure()

        # Plot all loss data
        plt.plot(loss_index.flatten(), vals.flatten(), color='blue')

        plt.xlabel('Number of training examples seen')
        plt.ylabel('loss')
        plt.title('{}'.format(self.algorithm))

        fig.show()
        fig_name = '{}/Results/train_loss{}.png'.format(self.algorithm, name)
        fig.savefig(fig_name)

    def confusion(self):
        # Load the data
        actual_images = np.load("MNIST/test_labels.npy")
        predicted_images = np.load(self.prediction_values_directory)
        # This function prints stuff in the alphabetical order
        # print the confusion matrix
        print(metrics.confusion_matrix(actual_images, predicted_images))
        # print the precision and recall, amount other metric
        print(metrics.classification_report(actual_images, predicted_images, digits=3))

    def runthrough(self, name=''):
        # Code to run during demo:
        self.load_data()
        self.train()
        self.test()
        self.evaluate(name)
        self.confusion()
