import random
import cv2
import numpy as np
from gendataset import Dataset
from util import rgb2bayer, areaOfGate
import tensorflow as tf
import os
# dataloader class
class AugmentedDataset(Dataset):
 
    # constructor
    def __init__(self, image_dir, csv_file, input_shape, output_shape, augment_methods):
        super().__init__(image_dir, csv_file, input_shape, output_shape)
        self.augment_methods = augment_methods
 
    def _loadImage(self, filename):
        """a function to load, augment and preprocess the images

        Args:
            filename (string): image directory

        Returns:
            numpy array: image
        """     
        image_file = tf.io.read_file(filename)
        image = tf.io.decode_png(image_file, channels = 3)
        image = tf.image.resize(image, self.input_shape[:2])
        if 'HSV' in self.augment_methods:
            image = tf.image.random_hue(image, 0.5)
            image = tf.image.random_saturation(image, 5, 10)
            image = tf.image.random_brightness(image, 5)
        if 'BlurGaussian' in self.augment_methods:
            if random.random() > 0.5:
                image = self.apply_blur(image)
        
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32)
        return image
    
    def _gaussian_kernel(self, kernel_size, sigma, n_channels, dtype):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


    def apply_blur(self, img):
        blur = self._gaussian_kernel(3, 2, 3, img.dtype)
        img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
        return img[0]
    
    def applyRotation(self, image, label):
        """a function to apply random rotation to the image and label

        Args:
            image (numpy array): image
            label (numpy array): label

        Returns:
            numpy array: image
            numpy array: label
        """        
        angle = random.randint(-15, 15)
        image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle)
        label = self.rotateLabel(label, angle)
        return image, label
if __name__=='__main__':
    image_dir = 'dataset/CNN/Austin1'
    csv_file = 'dataset/CNN/Austin1/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (120, 280, 1)#height, width, channels
    output_shape = (4,)


    # Iniltiaize Class
    dataset = AugmentedDataset(image_dir, csv_file, input_shape, output_shape, ['HSV', 'BlurGaussian'])

    # Create the dataset
    train_dataset, val_dataset = dataset.createDataset(batch_size=30)

    #Visualize the dataset
    batch = next(iter(train_dataset))
    images, labels = batch

    # Plot the images in the batch to check
    for i in range(len(images)):
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_GRAY2BGR)
        for j in range(0, len(labels[i]), 2):
            cv2.circle(image, (int(labels[i][j]), int(labels[i][j+1])), 10, (0, 255, 0), -1)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print('done')