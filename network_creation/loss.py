import tensorflow as tf
import numpy as np
import tensorflow as tf

def loss(y_true, y_pred):
    #hyperparameters
    lambda_coord = 1
    lambda_noobj = 2

    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    confidence_true = y_true[:, 0:1]
    confidence_pred = y_pred[:, 0:1]

    center_coord_true = y_true[:, 1:3]
    center_coord_pred = y_pred[:, 1:3]
    
    size_true = y_true[:, 3:5]
    size_pred = y_pred[:, 3:5]

    xy_loss = tf.multiply(confidence_true, tf.math.squared_difference(center_coord_true, center_coord_pred))
    size_loss = tf.multiply(confidence_true, tf.math.squared_difference(size_true, size_pred))

    confidence_loss =  tf.multiply(confidence_true,tf.math.squared_difference(confidence_true, confidence_pred))
    nobj = 1 - confidence_true
    nobj_loss = tf.multiply(nobj, tf.math.squared_difference(confidence_true, confidence_pred))

    total_loss = lambda_coord*(tf.reduce_sum(xy_loss) + tf.reduce_sum(size_loss)) + lambda_noobj*(confidence_loss + nobj_loss)

    return total_loss

def normalizedLoss(height,width):
    def lossNorm(y_true, y_pred):
        #hyperparameters
        lambda_coord = 1
        lambda_noobj = 2

        norm_vec = [1/width,1/height]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        confidence_true = y_true[:, 0:1]
        confidence_pred = y_pred[:, 0:1]

        center_coord_true = y_true[:, 1:3]*norm_vec
        center_coord_pred = y_pred[:, 1:3]*norm_vec
        
        size_true = y_true[:, 3:5]*norm_vec
        size_pred = y_pred[:, 3:5]*norm_vec

        xy_loss = tf.multiply(confidence_true, tf.math.squared_difference(center_coord_true, center_coord_pred))
        size_loss = tf.multiply(confidence_true, tf.math.squared_difference(size_true, size_pred))

        confidence_loss =  tf.multiply(confidence_true,tf.math.squared_difference(confidence_true, confidence_pred))
        nobj = 1 - confidence_true
        nobj_loss = tf.multiply(nobj, tf.math.squared_difference(confidence_true, confidence_pred))

        total_loss = lambda_coord*(tf.reduce_sum(xy_loss) + tf.reduce_sum(size_loss)) + lambda_noobj*(confidence_loss + nobj_loss)

        return total_loss
    return lossNorm

def errorSizeX(y_true,y_pred):
    size_true = y_true[:, 3:5]
    size_pred = y_pred[:, 3:5]

    size_diffX = size_true[0] - size_pred[0]

    return size_diffX

def errorSizeY(y_true,y_pred):
    size_true = y_true[:, 3:5]
    size_pred = y_pred[:, 3:5]

    size_diffY = size_true[1] - size_pred[1]

    return size_diffY

def errorCenter(y_true,y_pred):
    center_coord_true = y_true[:, 1:3]
    center_coord_pred = y_pred[:, 1:3]
    
    center_error = abs(center_coord_pred[0]-center_coord_true[0])+abs(center_coord_pred[1]-center_coord_true[1])

    return center_error



if __name__ == '__main__':

    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(5)
    ])

    # Compile the model using the custom MSE loss function
    model.compile(optimizer='adam', loss=loss, metrics=['mse'])

    x_train = np.random.rand(1000, 10)  
    y_train = np.random.rand(1000, 5) 
    # Train the model
    model.fit(x_train, y_train, epochs=10)