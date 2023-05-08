import tensorflow as tf
import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# steps to create a dataset
# 1. create a dataset from the image directory and CSV file
#split image in 64, 64 quadrants
#check if corner is in quadrant
#if not mearsure distance and angle to top left corner or say to which quadrant to go next up down left right



class DatasetActive():
    def __init__(self, image_dir, csv_file, input_shape, output_shape, quadrant_size):
        """_summary_

        Args:
            image_dir (_type_): _description_
            csv_file (_type_): _description_
            input_shape (_type_): _description_
            output_shape (_type_): _description_
            quadrant_size (_type_): (width, height)
        """        
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.quadrant_size = quadrant_size


    def _loadCorners(self, filename, image_shape, corners_labels_df):
        row = corners_labels_df[corners_labels_df['filename'] == filename].iloc[0]
        corners = np.array([row['corner1_x'], row['corner1_y'], row['corner2_x'], row['corner2_y'], row['corner3_x'], row['corner3_y'], row['corner4_x'], row['corner4_y']])
       
        # Normalize the corners to input shape 
        corners = corners*np.array([self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0], self.input_shape[1]/image_shape[1], 
                                    self.input_shape[0]/image_shape[0], self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0], 
                                    self.input_shape[1]/image_shape[1], self.input_shape[0]/image_shape[0]])
        # TODO: what if there is more than gate in image? --> confidence value

        return corners.astype('float32')
    

    def preProcess(self, CNN_image_dir, CNN_csv_corners):
        """function to preprocess the image and corners

        Args:
            image (numpy array): image
            corners (numpy array): corners

        Returns:
            numpy array: image
            numpy array: corners
        """
        corners_labels_df = pd.read_csv(CNN_csv_corners)
        image_files = [file for file in os.listdir(CNN_image_dir) if file.endswith('.png')]

        # Detemrine number of quadrants
        num_rows = int(self.input_shape[0]/self.quadrant_size[0])
        num_cols= int(self.input_shape[1]/self.quadrant_size[1])

        #Reality check
    
        for i in image_files:
            image_org = cv2.imread( os.path.join( CNN_image_dir, i))
            
            image = cv2.resize(image_org, self.input_shape[:2][::-1])
            
            counter = 0
            for j in range(num_rows):
                for k in range(num_cols):
                    counter +=1
                    start_row = j * self.quadrant_size[0]
                    end_row = (j + 1) * self.quadrant_size[0]
                    start_col = k * self.quadrant_size[1]
                    end_col = (k + 1) * self.quadrant_size[1]

                    quadrant = image[start_row:end_row, start_col:end_col]

                    # Save quadrant
                    cv2.imwrite(os.path.join(self.image_dir, i[:-4] + '_' + str(counter) + '.png'), quadrant)

                    # cv2.imshow("quadrant", quadrant)
                    # cv2.imshow("Image", image)
                    # cv2.waitKey(0)
                    
                    # load top left corner
                    corners = self._loadCorners(i, image_org.shape, corners_labels_df)[:2]

                    # corner is in quadrant
                    confidence = 1
                    distance = 0
                    angle = 0
                    if (corners[0]<start_col or corners[0]>end_col or corners[1]<start_row or corners[1]>end_row):
                        #calculate distance and angle to top left corner from center of quadrant
                        confidence = 0
                        center = np.array([start_col + self.quadrant_size[1]/2, start_row + self.quadrant_size[0]/2])
                        distance = np.linalg.norm(corners - center)
                        angle = np.arctan2(corners[1] - center[1], corners[0] - center[0])

                    #make entry in csv file
                    with open(self.csv_file, 'a') as f:
                        f.write(i[:-4] + '_' + str(counter) + '.png' + ',' + str(confidence) + ',' + str(distance) + ',' + str(angle) + '\n')
        
        print('Preprocessing done')
                    

    def _loadImage(self, filename):
        """a function to load and preprocess the images

        Args:
            filename (string): image directory

        Returns:
            numpy array: image
        """ 
        image_dir = tf.io.read_file(filename)
        image = tf.io.decode_png(image_dir, channels = 3)
        image = tf.cast(image, tf.float32)*(1/255)
        return image   

    

    def _loadLabel(self, filename):
        """function to load and preprocess the labels

        Args:
            filename (string): filename of the labels

        Returns:
            numpy array: confidence, distance[pixel], angle[rad]
        """ 

        row =  self.labels_df[ self.labels_df['filename'] == os.path.basename(filename.numpy().decode('utf-8'))].iloc[0]
        results = np.array([row['confidence'], row['distance'], row['angle']])
       
        return results.astype('float32')



    
    def createDataset(self, batch_size=32):
        """function to create a TensorFlow dataset from the image directory and CSV file

        Args:
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            dict: tarining dataset, validation dataset
        """        
        self.labels_df = pd.read_csv(self.csv_file)
        #convert radians to degrees from 0 to 360

        self.labels_df['angle'] = np.where(self.labels_df['angle']<0, self.labels_df['angle']+2*np.pi, self.labels_df['angle'])
        self.labels_df['angle'] = np.where(self.labels_df['angle']>2*np.pi, self.labels_df['angle']-2*np.pi, self.labels_df['angle'])
        self.labels_df['angle'] = self.labels_df['angle'] * 180 / np.pi
        # Get a list of all direcotry image filenames 
        image_files = []
        image_files = [os.path.join(self.image_dir,file) for file in os.listdir(self.image_dir) if file.endswith('.png')]
        
        # Create a TensorFlow dataset from the image filenames
        filenames_ds = tf.data.Dataset.from_tensor_slices(image_files)
        
        # Map the load_image and load_label functions to the filenames to create the final dataset
        image_ds = filenames_ds.map(self._loadImage, num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = filenames_ds.map(lambda x: tf.py_function(self._loadLabel, [x], tf.float32))

        #Create validation and training sets
        train_size = int(0.8 * len(image_files))
        train_dataset = tf.data.Dataset.zip((image_ds.take(train_size), label_ds.take(train_size)))
        val_dataset = tf.data.Dataset.zip((image_ds.skip(train_size), label_ds.skip(train_size)))

        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset






                    




if __name__ == "__main__":

    CNN_image_dir = 'dataset/CNN/Boston_Images'
    CNN_csv_corners = 'dataset/CNN/Boston_Images/corners.csv'

    image_dir = 'dataset/ActiveVision/Boston_quadrants'
    csv_file = 'dataset/ActiveVision/Boston_quadrants/Boston_quadrants.csv'
    dataset = DatasetActive(image_dir, csv_file, input_shape=(360, 600, 3), output_shape=(3), quadrant_size=(90, 150))
    #dataset.preProcess(CNN_image_dir, CNN_csv_corners) #only run once to create dataset
    train_dataset, val_dataset = dataset.createDataset(batch_size = 1)

    #Visualize the dataset
    batch = next(iter(val_dataset.take(3)))
    images, labels = batch

    # Plot the images in the batch to check
    for i in range(len(images)):
        image = cv2.cvtColor((images[i].numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if labels[i][0]== 1:
            image = cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)
        else:
            #add arrow to image to show distance and angle
            center = np.array([dataset.quadrant_size[1]/2, dataset.quadrant_size[0]/2])
            distance = labels[i][1]
            angle = labels[i][2]
            corner = center + np.array([distance*np.cos(angle)/8, distance*np.sin(angle)/8])
            image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner.astype('int')), (0, 0, 255), 2)
        cv2.imwrite("Image", image)
        cv2.waitKey(0)
    print('done')