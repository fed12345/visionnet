import sys
import cv2
import csv
import os
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


def trainNetwork(model_name, dataset_dir, csv_name, input_shape, output_shape, batch_size, epochs, epochs_optimization, save_model, device, aware_quantization, pruning):
    if model_name == 'activevision':
        createModel = createModelActiveVision
    elif model_name == 'visionnet':
        createModel = createModel
    elif model_name == 'dronet':
        createModel = createModelDronet
        if aware_quantization:
            print('Aware Quantization Not Supported for Dronet')
            exit()
    elif model_name == 'gatenet':
        createModel = createModelGateNet
    else:
        print('Invalid model name')
        exit()

    image_dirs =[files for files in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, files))]
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
        with open('evalutation/models_test/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_base.tflite', 'wb') as f:
            f.write(tflite_model)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('evalutation/Loss.png', format='png')
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
            history = model.fit(datasetTrain, epochs=epochs_optimization, validation_data=datasetVal, verbose = 1, callbacks=callbacks)
            #Remove pruning wrappers
            model = tfmot.sparsity.keras.strip_pruning(model)
            model.summary()
            pruning_accuracy = history.history['val_loss'][-1]
            # Convert the TensorFlow model to a TensorFlow Lite model

    if save_model:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
        tflite_model = converter.convert()

        # Save the TensorFlow Lite model to a file
        with open('evalutation/models/'+str(model_name)+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_pruned.tflite', 'wb') as f:
            f.write(tflite_model)

    with strategy.scope():
        if aware_quantization:

            model = tfmot.quantization.keras.quantize_model(model)
            
            model.compile(optimizer=opt, loss='mse')
            history = model.fit(datasetTrain, epochs=epochs_optimization, validation_data=datasetVal, verbose = 1)
            quantization_accuracy = history.history['val_loss'][-1]
            model.summary()

        if save_model:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Save the TensorFlow Lite model to a file
            with open('evalutation/models_test/'+model_name+'_'+ str(input_shape[1])+'x'+str(input_shape[0])+'_quant.tflite', 'wb') as f:
                f.write(tflite_model)
    # Visualize Predictions
    predictions = model.predict(datasetVal)
    imagesVal, labelsVal = next(iter(datasetVal))
    return predictions, imagesVal, labelsVal, baseline_accuracy

