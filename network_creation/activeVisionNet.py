import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

#Start simple by dividing into 4 quadrants and check if the gate is in the quadrant
#If it is, then the quadrant is 1, else 0
#Then, extend output to 3 dimension (gate[0 or 1], distance to move for next quadrant, angle to move for next quadrant)

def createModelActiveVision(input_shape=(30, 45, 3)):
    # Define the model architecture
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape, name='input'),
        layers.Conv2D(16, (3,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3)),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(16, (3,3)),
        layers.BatchNormalization(),
        layers.Activation('linear'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='linear'),
        layers.Dense(3)
    ])

    # Compile the model with loss function, optimizer, and metrics l bvc;
    model.compile(optimizer='adam',
                loss='mse', #look into chamfer distance
                metrics=['mae', 'mse']) #look into new metrics
    
    return model

if __name__ == "__main__":
    model = createModelActiveVision()
    model.summary()
    