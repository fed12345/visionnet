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
            image = tf.image.random_hue(image, 0.1)
            image = tf.image.random_saturation(image, 0.5, 2)
            image = tf.image.random_brightness(image, 0.5)
        if 'BlurGaussian' in self.augment_methods:
            if random.random() > 0.5:
                image = self.apply_blur(image)
        
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
    
    def translationImage(self, image, label):
        # Use tf.py_function to wrap OpenCV translation function
        [translated_image, label] = tf.py_function(self._translate_and_crop, [image, label], [tf.float32, tf.float32])

        return translated_image, label

    def _translate_and_crop(self, image, label):
        # Convert tf.Tensor to numpy array
        image_np = image.numpy()
        tf.print(image_np.shape)
        tf.print(label)
        # Define the translation matrix
        tx, ty = np.random.randint(-10, 20, 2)

        trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Translate the image
        translated_image = cv2.warpAffine(image_np, trans_matrix, (image_np.shape[1], image_np.shape[0]), borderMode=cv2.BORDER_REPLICATE)

        resized_image = cv2.resize(translated_image, (self.input_shape[1], self.input_shape[0]))
        label = label.numpy()
        #fix label
        label[3] = label[3] + tx
        label[4] = label[4] + ty
        return resized_image, label       


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
        train_dataset = train_dataset.map(self.translationImage)
        # Batch the dataset
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        return train_dataset, val_dataset

if __name__=='__main__':
    image_dir = 'dataset/CNN/Mavlab_himax1'
    csv_file = 'dataset/CNN/Mavlab_himax1/corners.csv'
    tf.config.run_functions_eagerly(True)
    # Define the input and output shapes of the model
    input_shape = (120, 280, 3)#height, width, channels
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
        image = cv2.cvtColor((images[i].numpy().astype('float')).astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.circle(image, (int(labels[i][3]), int(labels[i][4])), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    print('done')