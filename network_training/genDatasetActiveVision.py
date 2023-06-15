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

def areaOfGate(corner1,corner2,corner3,corner4):
    """Compute area of gate

    Args:
        corner1 (tuple): x,y of corner
        corner2 (tuple): x,y of corner
        corner3 (tuple): x,y of corner
        corner4 (tuple): x,y of corner

    Returns:
        float: area of gate
    """    
    # compute the area of the gate
    # https://www.mathopenref.com/coordpolygonarea.html
    area = 0.5*np.abs((corner1[0]*corner2[1] + corner2[0]*corner3[1] + corner3[0]*corner4[1] + corner4[0]*corner1[1] - corner2[0]*corner1[1] - corner3[0]*corner2[1] - corner4[0]*corner3[1] - corner1[0]*corner4[1]))
    return area

def selectTopLeftCorner(corners):
    """Select the top leftcorner

    Args:
        corners (list): list of corners

    Returns:
        tuple: x,y of top left corner
    """
    #Find 2 conrens with smallest x
    corners.sort(key=lambda x:x[0])
    corner1 = corners[0]
    corner2 = corners[1]
    if corner1[1] < corner2[1]:
        return corner1
    else:
        return corner2

def find_corner_positions(gate_corners):
    # Sort the corners by their x-coordinate
    sorted_corners = sorted(gate_corners, key=lambda corner: corner[0])

    # Determine top left and top right corners based on y-coordinate
    if sorted_corners[0][1] < sorted_corners[1][1]:
        top_left = sorted_corners[0]
        top_right = sorted_corners[1]
    else:
        top_left = sorted_corners[1]
        top_right = sorted_corners[0]

    # Determine bottom left and bottom right corners based on y-coordinate
    if sorted_corners[2][1] < sorted_corners[3][1]:
        bottom_left = sorted_corners[2]
        bottom_right = sorted_corners[3]
    else:
        bottom_left = sorted_corners[3]
        bottom_right = sorted_corners[2]

    return [top_left, top_right, bottom_left, bottom_right]

    

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
        rows = corners_labels_df[corners_labels_df['filename'] == filename]
        #Choose row with biggest gate
        biggest_area = 0
        for i in range(len(rows)):
            row = rows.iloc[i]
            area = areaOfGate((row['corner1_x'], row['corner1_y']), (row['corner2_x'], row['corner2_y']), (row['corner3_x'], row['corner3_y']), (row['corner4_x'], row['corner4_y']))
            if area > biggest_area:
                biggest_area = area
                biggest_row = row
    


        
        corners = [(biggest_row['corner1_x'], biggest_row['corner1_y']), (biggest_row['corner2_x'], biggest_row['corner2_y']), (biggest_row['corner3_x'], biggest_row['corner3_y']), (biggest_row['corner4_x'], biggest_row['corner4_y'])]
        multiplied_corners = []
        for corner in corners:
            multiplied_corner = (corner[0] *self.input_shape[1]/image_shape[1], corner[1] * self.input_shape[1]/image_shape[1])
            multiplied_corners.append(multiplied_corner)
        return multiplied_corners
    

    def preProcess(self, CNN_image_dir, CNN_csv_corners):
        """function to preprocess the image and corners

        Args:
            image (numpy array): image
            corners (numpy array): corners

        Returns:
            numpy array: image
            numpy array: corners
        """
        with open(self.csv_file, 'a') as f:
            f.write('filename' + ',' + 'top_left' + ',' + 'top_right' + ',' + 'bottom_left' + ',' + 'bottom_right' +  '\n')

        corners_labels_df = pd.read_csv(CNN_csv_corners)
        image_files = [file for file in os.listdir(CNN_image_dir) if file.endswith('.png')]

        #Reality check
        if not (self.input_shape[0]%self.quadrant_size[0] == 0 or self.input_shape[1]%self.quadrant_size[1] == 0):
            raise ValueError('Input shape must be divisible by quadrant size')
        
        # Detemrine number of quadrants
        num_rows = int(self.input_shape[0]/self.quadrant_size[0])
        num_cols= int(self.input_shape[1]/self.quadrant_size[1])
    
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
                    corners = self._loadCorners(i, image_org.shape, corners_labels_df)

                    sorted_corners = find_corner_positions(corners)
                    confidence = [0,0,0,0]
                    for l in range(len(sorted_corners)):
                        col, row = sorted_corners[l]
                        if start_row <= row <= end_row and start_col <= col <= end_col:
                            confidence[l] = 1

                    
                    #make entry in csv file
                    with open(self.csv_file, 'a') as f:
                        f.write(i[:-4] + '_' + str(counter) + '.png' + ',' + str(confidence[0]) + ',' + str(confidence[1]) + ',' + str(confidence[2]) + ',' + str(confidence[3]) + '\n')
        
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
            numpy array: confidence, corner_x[pixel],corner_y[pixel]
        """ 

        row =  self.labels_df[ self.labels_df['filename'] == os.path.basename(filename.numpy().decode('utf-8'))].iloc[0]
        results = np.array([row['top_left'], row['top_right'], row['bottom_left'], row['bottom_right']])
       
        return results.astype('int8')



    
    def createDataset(self, batch_size=32):
        """function to create a TensorFlow dataset from the image directory and CSV file

        Args:
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            dict: tarining dataset, validation dataset
        """        
        self.labels_df = pd.read_csv(self.csv_file)
        # Get a list of all direcotry image filenames 
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
        #Shuffle
        #train_dataset = train_dataset.shuffle(buffer_size=train_size)
        #val_dataset = val_dataset.shuffle(buffer_size=len(image_files) - train_size)
        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset





def reconsturctImage(csv, image = 'img_1_'):
    """Reconstructs the image from the quadrants and add arrows to corner based on csv

    Args:
        csv (string): csv file with confidence and corner location
        image (str, optional): _description_. Defaults to 'img_1'.
    """  
    #Find all quadrants for image
    quadrants = [f for f in os.listdir('dataset/ActiveVision/Austin1_quadrants') if f.startswith(image)]
    #Sort quadrants based on number after img_1_
    quadrants.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    #Read data form csv file
    df = pd.read_csv(csv)
    quadrant_dict = {'name':[], 'confidence':[], 'corner_x':[], 'corner_y':[], 'center_x':[], 'center_y':[]}
    for quadrant in quadrants:
        #Find confidence and corner location
        row = df[df['filename'] == quadrant]
        quadrant_dict['name'].append(quadrant)
        quadrant_dict['confidence'].append(row['confidence'].values[0])
        quadrant_dict['corner_x'].append(row['corner_x'].values[0])
        quadrant_dict['corner_y'].append(row['corner_y'].values[0])

    #Find Center of quadrant large image
    shape = (4,4)
    for i in range(shape[0]):
        start_row = i*90
        for j in range(shape[1]):
            start_col = j*150
            quadrant_dict['center_x'].append(start_col+75)
            quadrant_dict['center_y'].append(start_row+45)
    

    # Initialize variables
    image_rows = []
    row_lengths = [4, 4, 4, 4]  # Each row contains 4 images
    start_idx = 0
    # Create rows
    for row_len in row_lengths:
        row = cv2.imread(os.path.join('dataset/ActiveVision/Austin2_quadrants', quadrant_dict['name'][start_idx]))
        for i in range(start_idx + 1, start_idx + row_len):
            img_arrow = cv2.imread(os.path.join('dataset/ActiveVision/Austin2_quadrants', quadrant_dict['name'][i]))
            row = np.hstack((row, img_arrow ))
        image_rows.append(row)
        start_idx += row_len


    image = np.vstack(tuple(image_rows))
    #Loop through all quadrants and add arrows from the center of each quadrant
    for i in range(len(quadrant_dict['name'])):
        if quadrant_dict['confidence'][i] == 1:
            continue
        #Add Arrow from center of quadrant to corner
        image = cv2.arrowedLine(image, (int(quadrant_dict['center_x'][i]), int(quadrant_dict['center_y'][i])), (int(quadrant_dict['corner_x'][i]+quadrant_dict['center_x'][i]), 
                                                                                                                int(quadrant_dict['corner_y'][i]+quadrant_dict['center_y'][i])), 
                                                                                                                (0,0,255), 1, tipLength=0.01)
  
    cv2.imshow('Image',image)
    cv2.waitKey(0)



if __name__ == "__main__":

    CNN_image_dir = 'dataset/CNN/'

    image_dir = 'dataset/ActiveVisionClassification/'
    #list all directories in CNN_image_dir
    CNN_image_dirs = [file for file in os.listdir(CNN_image_dir) if os.path.isdir(os.path.join(CNN_image_dir, file))]
    for CNN_images in CNN_image_dirs:
        #Get csv file for each directory
        CNN_image_dir_internal = os.path.join(CNN_image_dir, CNN_images)
        CNN_csv_corners = os.path.join(CNN_image_dir, CNN_images, 'corners.csv')
        csv_file = os.path.join(image_dir, CNN_images, 'data.csv')
        image_file = os.path.join(image_dir, CNN_images)
        if not os.path.isdir(image_file):
            os.makedirs(image_file)
        #Create dataset for each directory
        dataset = DatasetActive(image_file, csv_file, input_shape=(120, 180, 3), output_shape=(3), quadrant_size=(40, 60))
        dataset.preProcess(CNN_image_dir_internal, CNN_csv_corners)


    train_dataset, val_dataset = dataset.createDataset(batch_size = 16)

    #Visualize the dataset
    batch = next(iter(train_dataset))
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
            corner_x = labels[i][1]
            corner_y = labels[i][2]
            corner = center + np.array([corner_x, corner_y])
            image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(center.astype('int')+corner.astype('int')), (0, 0, 255), 2)
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

    
    print('done')