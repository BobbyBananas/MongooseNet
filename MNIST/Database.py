import matplotlib.pyplot as plt
import numpy as np
import gzip


class Dataset:
    def __init__(self, img_location, label_location):
        # Save filenames
        self.img_location = img_location
        self.label_location = label_location

        # Initialise the decompressed data files
        self.img_file = 0
        self.label_file = 0

        # Initialise Dataset Information
        self.size = 0
        self.rows = 0
        self.cols = 0

        # Initialise the Usable Image and label Set
        self.set_img = 0
        self.set_label = 0

        self.pos_label = 0

    def load_data(self, size, train=bool):
        # Load, read, format and save the dataset
        self.size = size

        self.decompress_data()

        self.read_img()
        self.read_label()

        self.format_img()
        self.format_label()

        self.save_data(train)

    def decompress_data(self):
        # Unzip the gzip files
        self.img_file = gzip.open(self.img_location, 'r')
        self.label_file = gzip.open(self.label_location, 'r')

    def read_img(self):
        # MNIST Data format is a file of  unsigned bytes
        # The first 16 bytes give info about the file:
        magic = self.img_file.read(4)
        size = self.img_file.read(4) #Bad format, doesnt work
        rows = self.img_file.read(4)
        cols = self.img_file.read(4)

        # Extrapolate and store dataset information
        self.rows = rows[3]  # Only the final byte contains useful information
        self.cols = cols[3]

        # The Rest of the file has the number dataset
        self.set_img = self.img_file.read(self.rows * self.cols * self.size)

    def read_label(self):
        # MNIST Data format is a file of unsigned bytes
        # The first 8 bytes give info about the file:
        magic = self.label_file.read(4)
        labels = self.label_file.read(4)

        # The Rest of the file has the label dataset
        self.set_label = self.label_file.read(self.size)
        
    def format_img(self):
        # Convert the dataset numbers:
        #   Current Format => Bytes [b'\x00\x00\x00\x1c.....']
        #   Ideal Format   => Numpy Arrays (28x28x1)
        #                     Elements of unsigned 8bit integers [0 to 255]

        # Convert datatype from Bytes to uint8.
        data = np.frombuffer(self.set_img, dtype=np.uint8).astype(float)
        data = data/255
        # Convert shape from 1 dimension to 3 dimension
        # X many 28x28 vectors
        data = data.reshape(self.size, 1, self.rows, self.cols)

        # Save the formatted numbers
        self.set_img = data

    def format_label(self):
        # Convert the dataset labels:
        #   Current Format => Bytes [b'\x00\x00\x00\x1c.....']
        #   Ideal Format   => Numpy Arrays (size x 1)
        #                     Elements of

        # Convert datatype from Bytes to uint8.
        label_data = np.frombuffer(self.set_label, dtype=np.uint8)

        self.set_label = label_data

        # To get a positional array of the value ie[10^val]
        # Reshape array to two dimensions
        label_data = label_data.reshape(self.size, 1)

        # Transform the map the number to a spot in 10 bit binary vector
        ten_array = np.zeros((self.size, 10, 1), dtype=np.long)

        for i in range(0, self.size):
            ten_array[i, label_data[i]] = 1

        # Save the formatted labels
        self.pos_label = ten_array

    def save_data(self, train=bool):
        # Save our img and file data externally as numpy arrays.
        # Saved as 'train/test_images/labels.npy' in the MNIST folder

        if train:
            name = 'train'
        else:
            name = 'test'

        image_filename = 'MNIST/' + name + '_images'
        label_filename = 'MNIST/' + name + '_labels'

        np.save(image_filename, self.set_img)
        np.save(label_filename, self.set_label)

    def disp_img(self, from_img, to_img):
        # Display the greyscale images in the specified range
        for i in range(from_img, to_img):
            plt.imshow(self.set_img[i, 0], cmap='gray')
            plt.show()

    def disp_label(self, from_label, to_label):
        # Print the image labels in the specified range
        for i in range(from_label, to_label):
            print(self.set_label[i])


def initialise_data(train_size, test_size):
    print("Dataset Initialisation Phase")
    # Choose the dataset and our desired data size
    train_data = Dataset('MNIST/Compressed/train_images.gz', 'MNIST/Compressed/train_labels.gz')
    test_data = Dataset('MNIST/Compressed/test_images.gz', 'MNIST/Compressed/test_labels.gz')
    # Format and save the dataset for our desired size
    train_data.load_data(train_size, True)
    test_data.load_data(test_size, False)