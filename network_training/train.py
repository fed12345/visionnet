
import sys
import cv2
import csv
import os
import copy

sys.path.append("/Users/Federico/Desktop/Thesis/code/visionnet/network_creation")
sys.path.append("/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR")

from visionnet import createModel
from gendataset import Dataset
import dataset

def visualizePerdiction(image, prediction, actual):
    image = cv2.cvtColor((image.numpy().astype('float')*255).astype('uint8'), cv2.COLOR_RGB2BGR)
    for i in range(0, len(prediction), 2):
        cv2.circle(image, (int(prediction[i]), int(prediction[i+1])), 10, (0, 0, 255), -1)
        cv2.circle(image, (int(actual[i]), int(actual[i+1])), 10, (0, 255, 0), -1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

if __name__ == "__main__":
   
    # Define the image directory and CSV file
    imageDirAustin1 = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin1/'
    csvFileAustin1 = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin1/corners.csv'

    imageDirAustin2 = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin2/'
    csvFileAustin2 = '/Users/Federico/Desktop/Thesis/code/MAVLAB_GATES/data/AIRR/Austin2/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (600, 360, 3)
    output_shape = (8,) 
    
    # Initialize the datasets
    datasetAustin1 = Dataset(imageDirAustin1, csvFileAustin1, input_shape, output_shape)
    datasetAustin2 = Dataset(imageDirAustin2, csvFileAustin2, input_shape, output_shape)

    #Generate the datasets
    trainAustin1, valAustin1 = datasetAustin1.createDataset()
    trainAustin2, valAustin2 = datasetAustin2.createDataset()

    #Combine the datasets
    datasetTrain = trainAustin1.concatenate(trainAustin2)
    datasetVal = valAustin1.concatenate(valAustin2)
    print('Datasets Ready')

    # Create the model
    model = createModel()
    model.summary()

    # Train the model
    model.fit(datasetTrain, epochs=15, validation_data=datasetVal, verbose = 1)

    # Evaluate the model on the validation set
    valLoss, valAcc, _ = model.evaluate(datasetVal)
    print('Validation loss:', valLoss)
    print('Validation accuracy:', valAcc)

    #Visualize the Predictions
    predictions = model.predict(datasetVal)
    imagesVal, labelsVal = next(iter(datasetVal))
    for i in range(len(imagesVal)):
        visualizePerdiction(imagesVal[i], predictions[i], labelsVal[i])

