import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from util import areaOfGate, rgb2bayer, coord_out_of_bounds

class Dataset:
    def __init__(self, image_dir, csv_file, input_shape, output_shape):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Load the CSV file into a Pandas dataframe
        self.labels_df = pd.read_csv(csv_file)


    def _loadImage(self, filename):
        """a function to load and preprocess the images

        Args:
            filename (string): image directory

        Returns:
            numpy array: image
        """        
        image_file = tf.io.read_file(filename)
        image = tf.io.decode_png(image_file, channels = 3)
        image = tf.image.resize(image, self.input_shape[:2])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32)

        return image   

    

    def _loadLabel(self, filename):
        """function to load and preprocess the labels

        Args:
            filename (string): filename of the labels

        Returns:
            numpy array: corners
        """ 

        rows =  self.labels_df[self.labels_df['filename'] == os.path.basename(filename.numpy().decode('utf-8'))]
        assert len(rows) > 0, filename.numpy().decode('utf-8')
        biggest_area = -1
        for i in range(len(rows)):
            row = rows.iloc[i]
            area = areaOfGate((row['corner1_x'], row['corner1_y']), (row['corner2_x'], row['corner2_y']), (row['corner3_x'], row['corner3_y']), (row['corner4_x'], row['corner4_y']))
            if area > biggest_area:
                biggest_area = area
                biggest_row = row
            
        corners = [[biggest_row['corner1_x'], biggest_row['corner1_y']], [biggest_row['corner2_x'], biggest_row['corner2_y']], [biggest_row['corner3_x'], biggest_row['corner3_y']], [biggest_row['corner4_x'], biggest_row['corner4_y']]]
       
        #Determine size of dataset image
        image_shape = (cv2.imread(filename.numpy().decode('utf-8'))).shape
        confidence = np.ones(4)
        for i in range(len(corners)):
            if coord_out_of_bounds(image_shape[1], image_shape[0],corners[i][0],corners[i][1]) or (corners[i][0] == 0 and  corners[i][1] == 0):
                corners[i][0] = 0
                corners[i][1] = 0
                confidence[i] = 0
        confidence *= 100
        # Normalize the corners to input shape 
        corners = np.array(corners).flatten()*np.array([self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0], self.input_shape[1]/image_shape[1], 
                                    self.input_shape[0]/image_shape[0], self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0], 
                                    self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0]])

        return np.append(corners, confidence).astype('float32')



    
    def createDataset(self, batch_size=32):
        """function to create a TensorFlow dataset from the image directory and CSV file

        Args:
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            dict: tarining dataset, validation dataset
        """        

        # Get a list of all image filenames
        image_files = [os.path.join(self.image_dir,file) for file in os.listdir(self.image_dir) if file.endswith('.png')]
        
       
        # Create a TensorFlow dataset from the image filenames
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_files)

        # Map the load_image and load_label functions to the filenames to create the final dataset
        image_ds = filenames_ds.map(self._loadImage,num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = filenames_ds.map(lambda x: tf.py_function(self._loadLabel, [x], tf.float32))

        #Create validation and training sets
        train_size = int(0.95 * len(image_files))
        train_dataset = tf.data.Dataset.zip((image_ds.take(train_size), label_ds.take(train_size)))
        val_dataset = tf.data.Dataset.zip((image_ds.skip(train_size), label_ds.skip(train_size)))

        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset


if __name__ == '__main__':

    # Define the image directory and CSV file
    image_dir = 'dataset/CNN/LittletonHQ'
    csv_file = 'dataset/CNN/LittletonHQ/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (120, 280, 3)#height, width, channels
    output_shape = (4,)


    # Iniltiaize Class
    dataset = Dataset(image_dir, csv_file, input_shape, output_shape)

    # Create the dataset
    train_dataset, val_dataset = dataset.createDataset()

    #Visualize the dataset
    batch = next(iter(val_dataset))
    images, labels = batch

    # Plot the images in the batch to check
    for i in range(len(images)):
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_GRAY2RGB)
        for j in range(0, len(labels[i])-1, 2):
            cv2.circle(image, (int(labels[i][j]), int(labels[i][j+1])), 10, (0, 255, 0), -1)
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
    print('done')