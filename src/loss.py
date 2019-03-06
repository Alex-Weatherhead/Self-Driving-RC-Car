from keras import backend as K
import tensorflow as tf

def rmse(y_true, y_pred):
    
    y_true = tf.Print(y_true, [y_true], message="y_true: ")
    y_pred = tf.Print(y_pred, [y_pred], message="y_pred: ")
    
    loss = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    return loss
    
def mse(y_true, y_pred):
    
    y_true = tf.Print(y_true, [y_true], message="y_true: ")
    y_pred = tf.Print(y_pred, [y_pred], message="y_pred: ")
    
    loss = K.mean(K.square(y_pred - y_true), axis=-1)

    return loss