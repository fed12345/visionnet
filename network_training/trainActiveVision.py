import sys
import cv2
import csv
import os
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("network_creation")

from activeVisionNet import createModelActiveVision
from genDatasetActiveVision import DatasetActive

device = '/device:GPU:2'
# Define the input and output shapes of the model
input_shape = (90,150,3) #quadrant size height, width, channels
output_shape = (3,)
batch_size = 10

#Define image directory
dataset_dir = 'dataset/ActiveVision/'

image_dirs = [files for files in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, files))]
datasetTrain = None
datasetVal = None
for image_dir in image_dirs:

    #Define csv file
    csv_file = os.path.join(dataset_dir,image_dir,'data.csv')

    #initialize the dataset
    dataset = DatasetActive(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape, input_shape)

    #Generate the datasets
    train, val = dataset.createDataset(batch_size=batch_size)

    #Combine the datasets
    if datasetTrain == None:
        datasetTrain = train
        datasetVal = val
    else:
        datasetTrain = datasetTrain.concatenate(train)
        datasetVal = datasetVal.concatenate(val)

print('Datasets Ready')


with tf.device(device):
    # Create the model
    model = createModelActiveVision(input_shape=input_shape)
    model.summary()
    # Train the model
    history = model.fit(datasetTrain, epochs=100, validation_data=datasetVal, verbose = 1)

#Plot loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('evalutation/activevision/Loss.png', format='png')
# Visualize Predictions
predictions = model.predict(datasetVal)
imagesVal, labelsVal = next(iter(datasetVal))
for i in range(len(labelsVal)):
    image = cv2.cvtColor((imagesVal[i].numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    if predictions[i][0] > 0.9:
        image = cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)
    else:
        #add arrow to image to show distance and angle
        center = np.array([input_shape[1]/2, input_shape[0]/2])

        corner = np.array([predictions[i][1]*150,  predictions[i][2]*150])
        corner = center + corner
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner.astype('int')), (255, 0, 0), 2)

        #add actual distance and angle
        corner_val = center + np.array([labelsVal[i][1]*150, labelsVal[i][2]*150])
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner_val.astype('int')), (0, 0, 255), 2)
    cv2.imwrite('evalutation/activevision/result'+ str(i) + '.png',image)

# Convert the TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('evalutation/models/activeVision.tflite', 'wb') as f:
    f.write(tflite_model)

print('done')

