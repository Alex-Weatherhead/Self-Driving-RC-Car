__author__ = "Alex Weatherhead"
__version__ = "0.0.0"

from keras import Model
from keras.backend import tanh
from keras.engine.topology import Input
from keras.layers.core import Dense, Dropout, Flatten, Activation, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization

def networkA(input_shape):
    
    i = Input(input_shape)
    x = Lambda(lambda x: x/127.5 - 1.0)(i)
    
    x = Conv2D(16, (1,1), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(16, (1,1), strides=(2,2), activation='elu', border_mode="valid")(x)
    
    x = Conv2D(32, (1,1), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(32, (1,1), strides=(2,2), activation='elu', border_mode="valid")(x)
    
    flatten = Flatten()(x)
    x = Dropout(0.50)(flatten)
    
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    
    x = Dense(1)(x)

    model = Model(i, x)
    
    return model

def networkB(input_shape):
    
    i = Input(input_shape)
    x = Lambda(lambda x: x/127.5 - 1.0)(i)
    
    x = Conv2D(16, (3,3), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(16, (3,3), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(16, (3,3), strides=(1,1), activation='elu', border_mode="valid")(x)
    
    x = Conv2D(32, (3,3), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(32, (3,3), strides=(2,2), activation='elu', border_mode="valid")(x)
    x = Conv2D(32, (3,3), strides=(1,1), activation='elu', border_mode="valid")(x)
    
    flatten = Flatten()(x)
    x = Dropout(0.50)(flatten)
    
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)
    
    x = Dense(1)(x)

    model = Model(i, x)
    
    return model
    
def nvidia(input_shape):
    
    inputs = Input(input_shape)
    
    normalization = Lambda(lambda x: x/127.5 - 1.0)(inputs)
    
    x = Convolution2D(24, (5,5), strides=(2,2), border_mode="valid")(normalization)
    x = ELU()(x)
    
    x = Convolution2D(36, (5,5), strides=(2,2), border_mode="valid")(x)
    x = ELU()(x)
    
    x = Convolution2D(48, (5,5), strides=(2,2), border_mode="valid")(x)
    x = ELU()(x)
    
    x = Convolution2D(64, (3,3), border_mode="valid")(x)
    x = ELU()(x)
    
    x = Convolution2D(64, (3,3), border_mode="valid")(x)
    flatten = Flatten()(x)
    x = Dropout(0.50)(flatten)
    x = ELU()(x)
    
    x = Dense(100)(x)
    x = ELU()(x)
    
    x = Dense(50)(x)
    x = ELU()(x)
    
    x = Dense(10)(x)
    x = ELU()(x)
    
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model
