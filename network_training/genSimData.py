import random
import cv2
import numpy as np
from gendataset import Dataset
from util import rgb2bayer, areaOfGate
import tensorflow as tf
import os

class SimDataset(Dataset):
    def __init__(self, image_dir, csv_file, input_shape, output_shape):
        super().__init__(image_dir, csv_file, input_shape, output_shape)

    def _loadLabel(self,filename):
        row =  self.labels_df[self.labels_df['filename'] == os.path.basename(filename.numpy().decode('utf-8'))]

        image_shape = (cv2.imread(filename.numpy().decode('utf-8'))).shape
        center_x = row['center_x'] * self.input_shape[1]/image_shape[1]
        center_y = row['center_y'] * self.input_shape[0]/image_shape[0]

        size_x = row['size_x'] * self.input_shape[1]/image_shape[1]
        size_y = row['size_y'] * self.input_shape[0]/image_shape[0]

        return np.array([1, size_x, size_y, center_x, center_y]).astype('float32')
    


if __name__== "__main__":
    image_dir = 'dataset/CNN/Sim_images'
    csv_file = 'dataset/CNN/Sim_images/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (120, 160, 3)#height, width, channelsß
    output_shape = (4,)

    dataset = SimDataset(image_dir, csv_file, input_shape, output_shape)
    train_dataset, val_dataset = dataset.createDataset()
    #Visualize the dataset
    batch = next(iter(train_dataset))
    images, labels = batch

    for i in range(len(images)):
        label = labels[i].numpy().astype('int')
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (label[3]-label[1]//2, label[4]-label[2]//2), (label[3]+label[1]//2, label[4]+label[2]//2), (0, 255, 0), 2)
        cv2.imshow("img", image)
        cv2.waitKey(0)
