import tensorflow as tf
import numpy as np
import tensorflow as tf

def loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    coord_true = y_true[:, :8]
    coord_pred = y_pred[:, :8]
    confidence_true = y_true[:, 8:]
    confidence_pred = y_pred[:, 8:]

    mse_confidence = tf.math.squared_difference(confidence_pred, confidence_true)
    confidence = 0.01 * tf.repeat(confidence_true, repeats=2, axis=1)
    weighted_coord_pred = tf.multiply(coord_pred, confidence)

    mse_coord = tf.math.squared_difference(weighted_coord_pred,coord_true)

    total_loss = tf.math.reduce_mean(tf.concat([mse_confidence,mse_coord],1))

    return total_loss


if __name__ == '__main__':

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(12)
    ])

    # Compile the model using the custom MSE loss function
    model.compile(optimizer='adam', loss=loss, metrics=['mse'])

    x_train = np.random.rand(1000, 10)  
    y_train = np.random.rand(1000, 12) 
    # Train the model
    model.fit(x_train, y_train, epochs=10)