__author__ = "Alex Weatherhead"
__version__ = "0.0.0"

from keras.optimizers import Adam, SGD
from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping

from model import nvidia, networkA, networkB
from loss import rmse, mse
from preprocessing import preprocessor
from augmentation import augmenter
from utils import populate, split, balance, unpack
from visualize import distribution_hist, losses_plot
from keras_generator import KerasGenerator

INPUT_SHAPE = (80, 320, 3)
MINIMUM_ANGLE = 40
MAXIMUM_ANGLE = 140

TRAINING_SIZE = 0.70
VALIDATION_SIZE = 0.30

DOWNSAMPLE_THRESHOLD = 50
UPSAMPLE_THRESHOLD = 50

EPOCHS = 20
BATCH_SIZE = [1, 2, 4, 8, 16, 32]
LR = [0.1, 0.01, 0.001, 0.0001]
LOSS = rmse
PATIENCE = 3
MAX_QUEUE_SIZE = 32
SHUFFLE = True

directory = '../dataset/training/'

if __name__ == '__main__':
    
    model_name = input("Enter a name for your model: ")
    
    dataset = {angle: [] for angle in range(MINIMUM_ANGLE, MAXIMUM_ANGLE + 1)}
    populate(dataset, directory, '.png')
    
    training_dataset, validation_dataset = split(dataset,
                                                 training_size=TRAINING_SIZE,
                                                 validation_size=VALIDATION_SIZE)
    
    distribution_hist('../visualizations/distribution_pre-balance.png', training_dataset)

    balance(training_dataset,
            downsample_threshold=DOWNSAMPLE_THRESHOLD,
            upsample_threshold=UPSAMPLE_THRESHOLD)
            
    distribution_hist('../visualizations/distribution_post-balance.png', training_dataset)
    
    training_data = unpack(training_dataset)
    validation_data = unpack(validation_dataset)

    for batch_size in BATCH_SIZE:
        
        training_data_generator = KerasGenerator(training_data,
                                                 batch_size,
                                                 INPUT_SHAPE,
                                                 preprocessor,
                                                 augmenter=augmenter)
                                                             
        validation_data_generator = KerasGenerator(validation_data,
                                                   batch_size,
                                                   INPUT_SHAPE,
                                                   preprocessor)
        
        for lr in LR:
        
            model_details = '{}-{}.{}'.format(batch_size, lr, model_name)
        
            model = nvidia(INPUT_SHAPE)
                
            optimizer = Adam(lr=lr)
            model.compile(optimizer=optimizer, loss=LOSS)
                
            progbar_logger = ProgbarLogger(count_mode='steps')
            model_checkpoint = ModelCheckpoint('../models/' + model_details + '.{epoch:02d}-{val_loss:.2f}.hdf5')
            early_stopping = EarlyStopping(patience=PATIENCE)
                
            callbacks = [progbar_logger, model_checkpoint, early_stopping]
            history = model.fit_generator(training_data_generator,
                                          validation_data=validation_data_generator,
                                          callbacks=callbacks,
                                          max_queue_size=MAX_QUEUE_SIZE,
                                          epochs=EPOCHS,
                                          shuffle=SHUFFLE)
        
            losses_plot('../visualizations/' + model_details + '.png',
                        history.history['loss'],
                        history.history['val_loss'],
                        EPOCHS)
    
    