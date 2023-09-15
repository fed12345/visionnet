import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import sys
import os

sys.path.append("network_creation")

from loss import loss
from gendataset import Dataset

# Load your model from a directory
path_to_model = "./model_directory/your_model_name"
model = tf.keras.models.load_model(path_to_model)

# Freeze convolutional layers
for layer in model.layers:
    if isinstance(layer, Conv2D):
        layer.trainable = False


dataset_dir = "dataset/CNN"
image_dirs = ["Mavlab_himax1", "Mavlab_himax2"]
csv_name = "corners.csv"

for image_dir in image_dirs:

    #Define csv file
    csv_file = os.path.join(dataset_dir,image_dir,csv_name)

    #initialize the dataset
    dataset = Dataset(os.path.join(dataset_dir,image_dir), csv_file, [180,120,1], 12)

    #Generate the datasets
    train, val = dataset.createDataset(batch_size=16)

    #Combine the datasets
    if datasetTrain == None:
        datasetTrain = train
        datasetVal = val
    else:
        datasetTrain = datasetTrain.concatenate(train)
        datasetVal = datasetVal.concatenate(val)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss)


EPOCHS = 100
BATCH_SIZE = 32
model.fit(datasetTrain, validation_data=datasetVal)

model.save('evalutation/models/gatenetfinetune_180x120.keras')
# Convert the TensorFlow model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the TensorFlow Lite model to a file
with open('evalutation/models/gatenetfinetune_180x120_base.tflite', 'wb') as f:
    f.write(tflite_model)