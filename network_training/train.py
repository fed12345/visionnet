import sys
import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
import json
import argparse

sys.path.append("network_creation")
sys.path.append("visionnet/network/visionnet/network_creation/")

from networktraining import trainNetwork
from visionnet import createModel, createModelDronet, createModelGateNet
from gendataset import Dataset
from activeVisionNet import createModelActiveVision
from genDatasetActiveVision import DatasetActive

#==========================PARAMETERS===================================================================================
# Create an argument parser
parser = argparse.ArgumentParser(description='Process configuration file.')

# Add an argument for the config file
parser.add_argument('--config', '-c', required=True, help='Path to the config file')

# Parse the command-line arguments
args = parser.parse_args()

with open(args.config, 'r') as file:
    json_data = file.read()

# Parse the JSON data into a dictionary
parameters = json.loads(json_data)

# Access the parameters
model_name = parameters['model']
device = parameters['device']
input_shape = parameters['input_shape']
output_shape = parameters['output_shape']
batch_size = parameters['batch_size']
epochs = parameters['epochs']
epochs_optimization = parameters['epochs_optimization']
dataset_dir = parameters['dataset_dir']
dataset_dir_augmented =  parameters['dataset_dir_augmented']
datasets = parameters['datasets']
csv_name = parameters['csv_name']
aware_quantization = parameters['aware_quantization']
pruning = parameters['pruning']
save_model = parameters['save_model']

input_shapes = [(350,400,3),(280,320,3),(210,240,3),(input_shape),(112,128,3), (70,80,3)]
#==========================CODE=========================================================================================
#plot accuracies


def visualizePrediction(image, prediction, actual):
    image = cv2.cvtColor((image.numpy().astype('float')).astype('uint8'), cv2.COLOR_BGR2RGB)
    cv2.circle(image, (int(prediction[3]), int(prediction[4])), 10, (0, 0, 255), -1)

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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

accuracies = []
for input_shape in input_shapes:
    predictions, imagesVal, labelsVal, accuracy = trainNetwork(model_name, dataset_dir,  dataset_dir_augmented, datasets, csv_name, input_shape, output_shape, batch_size, epochs, epochs_optimization, save_model, device, aware_quantization, pruning)
    accuracies.append(accuracy)

#plot accuracies
plt.plot(input_shapes, accuracies)
plt.xlabel('input shape')
plt.ylabel('mse')
plt.savefig('evalutation/accuracies.png')

for i in range(len(labelsVal)):
     cv2.imwrite('evalutation/cnn1/'+ str(i) + '.png',visualizePrediction(imagesVal[i], predictions[i], labelsVal[i]))

print('done')

