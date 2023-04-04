import tensorflow as tf
from tensorflow.keras import layers


def createModel():
    # Define the model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(600, 360, 3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(8)
    ])

    # Compile the model with loss function, optimizer, and metrics
    model.compile(optimizer='adam',
                loss='mse', #look into chamfer distance
                metrics=['mae', 'mse']) #look into new metrics
    
    return model

