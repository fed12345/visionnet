import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import TensorBoard
import sys
import os

sys.path.append("network_creation")

from loss import loss
from gendataset import Dataset
from dataAugment import AugmentedDataset

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
# Load your model from a directory
path_to_model = "evalutation/models/gatenet_300x200.keras"
log_dir = "evalutation/logfine"
model = tf.keras.models.load_model(path_to_model, compile=False)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Freeze convolutional layers
for layer in model.layers:
    if layer.name in ['conv1', 'conv2']:
        layer.trainable = False


dataset_dir = "dataset/CNN"
image_dirs = ["Mavlab_himax1", "Mavlab_himax2"]
csv_name = "corners.csv"
input_shape = [200,300,3]
output_shape = 5
batch_size = 16
datasetTrain = None
datasetVal = None
for image_dir in image_dirs:

    #Define csv file
    csv_file = os.path.join(dataset_dir,image_dir,csv_name)

    #initialize the dataset
    dataset = Dataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape)
    augmenteddata = AugmentedDataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape, ['HSV', 'BlurGaussian'])
    augmenteddata1 = AugmentedDataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape, [])
    augmenteddata2 = AugmentedDataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape, [])
    augmenteddata3 = AugmentedDataset(os.path.join(dataset_dir,image_dir), csv_file, input_shape, output_shape, ['HSV', 'BlurGaussian'])

    #Generate the datasets
    train, val = dataset.createDataset(batch_size=batch_size)
    train_aug, val_aug = augmenteddata.createDataset(batch_size=batch_size)
    train_aug1, _ = augmenteddata1.createDataset(batch_size=batch_size)
    train_aug2, _ = augmenteddata2.createDataset(batch_size=batch_size)
    train_aug3, _ = augmenteddata3.createDataset(batch_size=batch_size)

    #Combine the datasets
    if datasetTrain == None: 
        datasetTrain = train
        datasetVal = val

    else:
        datasetTrain = datasetTrain.concatenate(train)
        datasetVal = datasetVal.concatenate(val)
        datasetTrain = datasetTrain.concatenate(train_aug)
        datasetTrain = datasetTrain.concatenate(train_aug1)
        datasetTrain = datasetTrain.concatenate(train_aug2)
        datasetTrain = datasetTrain.concatenate(train_aug3)
        datasetVal = datasetVal.concatenate(val)
        datasetVal = datasetVal.concatenate(val_aug)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss)


EPOCHS = 50
model.fit(datasetTrain, validation_data=datasetVal, epochs = EPOCHS, callbacks = [tensorboard_callback])

model.save('evalutation/models/gatenetfinetune_300x200.keras')
# Convert the TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the TensorFlow Lite model to a file
with open('evalutation/models/gatenetfinetune_300x200_base.tflite', 'wb') as f:
    f.write(tflite_model)