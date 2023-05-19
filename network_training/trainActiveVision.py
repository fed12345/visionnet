import sys
import cv2
import csv
import os
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

sys.path.append("network_creation")
sys.path.append("visionnet/network/visionnet/network_creation/")

from visionnet import createModel, createModelDronet, createModelGateNet
from gendataset import Dataset
from activeVisionNet import createModelActiveVision
from genDatasetActiveVision import DatasetActive


#==========================PARAMETERS===================================================================================
model = 'gatenet' #visionnet, dronet, gatenet, activevision
device = '/device:GPU:2'
input_shape = (180,120,3) #quadrant size height, width, channels
output_shape = (3,)
batch_size = 10
epochs = 1
dataset_dir = 'dataset/CNN/'
csv_name= 'corners.csv'
quantization = True
pruning = True
save_model = True

#==========================CODE=========================================================================================

def visualizePrediction(image, prediction, actual):
    image = cv2.cvtColor((image.numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(0, len(prediction), 2):
        cv2.circle(image, (int(prediction[i]), int(prediction[i+1])), 10, (0, 0, 255), -1)
        cv2.circle(image, (int(actual[i]), int(actual[i+1])), 10, (0, 255, 0), -1)
    return image

def visualizePredictionActiveVision(image, prediction, actual):
    image = cv2.cvtColor((image.numpy().astype('float')*255).astype('uint8'), cv2.COLOR_BGR2RGB)
    if prediction[0] > 0.9:
        image = cv2.applyColorMap(image, cv2.COLORMAP_SUMMER)
    else:
        #add arrow to image to show distance and angle
        center = np.array([input_shape[1]/2, input_shape[0]/2])

        corner = np.array([prediction[1]*150,  prediction[2]*150])
        corner = center + corner
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner.astype('int')), (255, 0, 0), 2)

        #add actual distance and angle
        corner_val = center + np.array([actual[1]*150, actual[2]*150])
        image = cv2.arrowedLine(image, tuple(center.astype('int')), tuple(corner_val.astype('int')), (0, 0, 255), 2)
    return image

if model == 'activevision':
    VisualizePrediction = visualizePredictionActiveVision
    createModel = createModelActiveVision
    Dataset = DatasetActive
elif model == 'visionnet':
    VisualizePrediction = visualizePrediction
    createModel = createModel
elif model == 'dronet':
    VisualizePrediction = visualizePrediction
    createModel = createModelDronet
elif model == 'gatenet':
    VisualizePrediction = visualizePrediction
    createModel = createModelGateNet
else:
    print('Invalid model')
    exit()

image_dirs = [files for files in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, files))]
datasetTrain = None
datasetVal = None
for image_dir in image_dirs:

    #Define csv file
    csv_file = os.path.join(dataset_dir,image_dir,csv_name)

    #initialize the dataset
    dataset = Dataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape)

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

strategy = tf.distribute.OneDeviceStrategy(device=device)
with strategy.scope():
    # Create the model
    model = createModel(input_shape=input_shape)
    model.summary()
    # Train the model
    history = model.fit(datasetTrain, epochs=epochs, validation_data=datasetVal, verbose = 1)

if save_model:
    # Convert the TensorFlow model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('evalutation/models/'+model+'_base.tflite', 'wb') as f:
        f.write(tflite_model)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('evalutation/activevision/Loss.png', format='png')
baseline_accuracy = history.history['val_loss'][-1]


with strategy.scope():
    if pruning:
        #Prune the model
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0, frequency=100)}
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
        model.summary()
        #Train the model
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=opt, loss='mse')
        history = model.fit(datasetTrain, epochs=1, validation_data=datasetVal, verbose = 1, callbacks=callbacks)
        #Remove pruning wrappers
        model = tfmot.sparsity.keras.strip_pruning(model)
        model.summary()
        pruning_accuracy = history.history['val_loss'][-1]
        # Convert the TensorFlow model to a TensorFlow Lite model

if save_model:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('evalutation/models/'+model+'_pruned.tflite', 'wb') as f:
        f.write(tflite_model)

with strategy.scope():
    if quantization:
        #Apply QAT
        quant_model = tfmot.quantization.keras.quantize_model(model)
        quant_model.compile(optimizer=opt, loss='mse')
        history = quant_model.fit(datasetTrain, epochs=1, validation_data=datasetVal, verbose = 1)
        quantization_accuracy = history.history['val_loss'][-1]
        model.summary()

if save_model:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open('evalutation/models/'+model+'_quant.tflite', 'wb') as f:
        f.write(tflite_model)
print('Baseline Accuracy: ' + str(baseline_accuracy))
print('Pruning Accuracy: ' + str(pruning_accuracy))
print('Quantization Accuracy: ' + str(quantization_accuracy))
# Visualize Predictions
predictions = model.predict(datasetVal)
imagesVal, labelsVal = next(iter(datasetVal))
for i in range(len(labelsVal)):
     cv2.imwrite('visionnet/network/visionnet/evalutation/'+ str(i) + '.png',visualizePrediction(imagesVal[i], predictions[i], labelsVal[i]))
print('done')

