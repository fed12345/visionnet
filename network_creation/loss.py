import tensorflow as tf
import numpy as np
import tensorflow as tf

def loss(y_true, y_pred):
    
    coord_true = y_true[:, :8]
    coord_pred = y_pred[:, :8]
    confidence_true = y_true[:, 8:]
    confidence_pred = y_pred[:, 8:]

    mse_confidence = tf.reduce_mean(tf.square(confidence_true - confidence_pred))

    confidence = 0.001 * tf.repeat(confidence_true, repeats=2, axis=1)

    weighted_coord_true = tf.multiply(coord_true, confidence)

    mse_coord = tf.reduce_mean(tf.square(weighted_coord_true - coord_pred))

    total_loss = mse_confidence + mse_coord

    return total_loss



if __name__ == '__main__':

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(12)
    ])

    # Compile the model using the custom MSE loss function
    model.compile(optimizer='adam', loss=loss)

    x_train = np.random.rand(1000, 10)  
    y_train = np.random.rand(1000, 12)   
    # Train the model
    model.fit(x_train, y_train, epochs=10)