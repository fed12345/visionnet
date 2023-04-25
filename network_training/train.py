
import sys
import cv2
import csv
import os
import copy
import tensorflow as tf

sys.path.append("visionnet/network/visionnet/network_creation/")

from visionnet import createModel, createModelDronet
from gendataset import Dataset

def visualizePerdiction(image, prediction, actual):
    image = cv2.cvtColor((image.numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(0, len(prediction), 2):
        cv2.circle(image, (int(prediction[i]), int(prediction[i+1])), 10, (0, 0, 255), -1)
        cv2.circle(image, (int(actual[i]), int(actual[i+1])), 10, (0, 255, 0), -1)
    return image

if __name__ == "__main__":
   
    # Define the image directory and CSV file
    imageDirAustin1 = '/data/federico/visionnet/training_data/Austin1'
    csvFileAustin1 = '/data/federico/visionnet/training_data/Austin1/corners.csv'

    imageDirAustin2 = '/data/federico/visionnet/training_data/Austin2'
    csvFileAustin2 = '/data/federico/visionnet/training_data/Austin2/corners.csv'

    # Define the input and output shapes of the model
    input_shape = (120,180,3) #height, width, channels
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

    strategy = tf.distribute.OneDeviceStrategy(device = '/GPU:2')
    # Create the model
   
    with strategy.scope():
        model = createModel()
        model.summary()
        #Train the model
        model.fit(datasetTrain, epochs=50, validation_data=datasetVal, verbose = 1)

    # Evaluate the model on the validation set
    valLoss, valAcc, _ = model.evaluate(datasetVal)
    print('Validation loss:', valLoss)
    print('Validation accuracy:', valAcc)
    #Visualize the Predictions
    predictions = model.predict(datasetVal)
    imagesVal, labelsVal = next(iter(datasetVal))
    for i in range(len(imagesVal)):
        cv2.imwrite('visionnet/network/visionnet/evalutation/'+ str(i) + '.png',visualizePerdiction(imagesVal[i], predictions[i], labelsVal[i]))

    # print('Saving Model')
    # tf.saved_model.save(model, 'visionnet/network/visionnet/evalutation/visionnet')

    # # Convert the model
    # converter = tf.lite.TFLiteConverter.from_saved_model('visionnet/network/visionnet/evalutation/visionnet') # path to the SavedModel directory
    # tflite_model = converter.convert()
    
    # # Save the model.
    # with open('visionnet.tflite', 'wb') as f:
    #     f.write(tflite_model)   
    
    # Save the model to tflite
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # print('Loaded model')
    # tflite_model = converter.convert()
    # open("visionnet.tflite", "wb").write(tflite_model)

