import sys
import cv2
import csv
import os
import copy
import tensorflow as tf
import numpy as np

sys.path.append("network_creation")

from activeVisionNet import createModelActiveVision
from genDatasetActiveVision import DatasetActive

imageDirAustin1 = 'dataset/ActiveVision/Austin1_quadrants/'
csvFileAustin1 = 'dataset/ActiveVision/Austin1_quadrants/Austin1_quadrants.csv'

imageDirAustin2 = 'dataset/ActiveVision/Austin2_quadrants/'
csvFileAustin2 = 'dataset/ActiveVision/Austin2_quadrants/Austin2_quadrants.csv'

imageDirBoston = 'dataset/ActiveVision/Boston_quadrants/'
csvFileBoston = 'dataset/ActiveVision/Boston_quadrants/Boston_quadrants.csv'

device = '/GPU:2'
# Define the input and output shapes of the model
input_shape = (90,150,3) #quadrant size height, width, channels
output_shape = (3,)
batch_size = 32

#initialize the datasets
datasetAustin1 = DatasetActive(imageDirAustin1, csvFileAustin1, input_shape, output_shape, input_shape)
datasetAustin2 = DatasetActive(imageDirAustin2, csvFileAustin2, input_shape, output_shape, input_shape)
datasetBoston = DatasetActive(imageDirBoston,csvFileBoston,input_shape,output_shape,input_shape)

#Generate the datasets
trainAustin1, valAustin1 = datasetAustin1.createDataset(batch_size=batch_size)
trainAustin2, valAustin2 = datasetAustin2.createDataset(batch_size=batch_size)
trainBoston, valBoston = datasetBoston.createDataset(batch_size=batch_size)

#Combine the datasets
datasetTrain = trainAustin1.concatenate(trainAustin2)
datasetTrain = datasetTrain.concatenate(trainBoston)
datasetVal = valAustin1.concatenate(valAustin2)
datasetVal = datasetVal.concatenate(valBoston)
print('Datasets Ready')

strategy = tf.distribute.OneDeviceStrategy(device = device)
    # Create the model
   
with strategy.scope():
    # Create the model
    model = createModelActiveVision(input_shape=input_shape)
    model.summary()

    # Train the model
    model.fit(datasetTrain, epochs=10, validation_data=datasetVal, verbose = 2)

# Evaluate the model on the validation set
valLoss, valAcc, _ = model.evaluate(datasetVal)
print('Validation loss:', valLoss)
print('Validation accuracy:', valAcc)

# Visualize Predictions
predictions = model.predict(datasetVal)
imagesVal, labelsVal = next(iter(datasetVal))
for i in range(len(predictions)):
    image = cv2.cvtColor((imagesVal[i].numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if predictions[i][0] > 0.9:
        image = cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)
    else:
        #add arrow to image to show distance and angle
        center = np.array([input_shape[1]/2, input_shape[0]/2])

        distance = predictions[i][1]
        angle = predictions[i][2]
        corner = center + np.array([distance*np.cos(angle)/8, distance*np.sin(angle)/8])
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner.astype('int')), (255, 0, 0), 2)

        #add actual distance and angle
        corner_val = center + np.array([labelsVal[i][1]*np.cos(labelsVal[i][2])/8, labelsVal[i][1]*np.sin(labelsVal[i][2])/8])
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner_val.astype('int')), (0, 0, 255), 2)
    cv2.imwrite('evalutation/activevision/result'+ str(i) + '.png',image)
print('done')

