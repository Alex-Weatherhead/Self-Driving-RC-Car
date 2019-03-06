import numpy as np

def preprocessor(image):

    image = np.expand_dims(image, axis=0)
    
    return image