import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, image_dir, csv_file, input_shape, output_shape):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Load the CSV file into a Pandas dataframe
        self.labels_df = pd.read_csv(csv_file)


    # Define a function to load and preprocess the images
    def _loadImage(self, filename):
        image = cv2.imread( os.path.join(self.image_dir, filename.numpy().decode('utf-8')))
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        return image


    # Define a function to load and preprocess the labels
    def _loadLabel(self, filename):
        row =  self.labels_df[ self.labels_df['filename'] == filename].iloc[0]
        corners = np.array([row['corner1_x'], row['corner1_y'], row['corner2_x'], row['corner2_y'], row['corner3_x'], row['corner3_y'], row['corner4_x'], row['corner4_y']])
        return corners.astype('float32')



    # Define a function to create a TensorFlow dataset from the image directory and CSV file
    def createDataset(self, batch_size=32):

        # Get a list of all image filenames
        image_files = [file for file in os.listdir(self.image_dir) if file.endswith('.png')]
        
        # Create a TensorFlow dataset from the image filenames
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_files)
        
        # Map the load_image and load_label functions to the filenames to create the final dataset
        image_ds = filenames_ds.map(lambda x: tf.py_function(self._loadImage, [x], tf.float32))
        label_ds = filenames_ds.map(lambda x: tf.py_function(self._loadLabel, [x], tf.float32))
        #Create validation and training sets
        train_size = int(0.8 * len(image_files))
        train_dataset = tf.data.Dataset.zip((image_ds.take(train_size), label_ds.take(train_size)))
        val_dataset = tf.data.Dataset.zip((image_ds.skip(train_size), label_ds.skip(train_size)))

        # Shuffle and batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset


if __name__ == '__main__':

    # Define the image directory and CSV file
    image_dir = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin1/'
    csv_file = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin1/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (600, 360, 3)
    output_shape = (4,)


    # Iniltiaize Class
    dataset = Dataset(image_dir, csv_file, input_shape, output_shape)

    # Create the dataset
    train_dataset, val_dataset = dataset.createDataset()

    #Visualize the dataset
    batch = next(iter(val_dataset.take(3)))
    images, labels = batch

    # Plot the images in the batch
    plt.figure(figsize=(10, 10))
    for i in range(7):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("float"))
        plt.axis("off")
    plt.show()
    print('done')