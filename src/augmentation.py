import random

import numpy as np
from PIL import Image

def augmenter(image, angle):

    if random.random() > 0.5:
        
        image, angle = flip(image, angle)
    
    if random.random() > 0.5:
        
        image = brighten(image)
        
    return image, angle

def flip(array, angle):
    """
    """
    new_array = array[:,::-1,:]
    new_angle = abs(140 - angle) + 40
    
    return new_array, new_angle
    
def shift(array, angle, min_shift=-160, max_shift=+160):
    """
    """
    new_array = np.zeros_like(array)
    new_angle = None
    
    tx = round(random.uniform(min_shift, max_shift))

    if tx < 0:
        crop = array[:,abs(tx):,:]
        new_array[:,:tx,:] = crop
        new_angle = np.clip((angle + angle * abs(tx)/320), 40, 140)
    elif tx > 0:
        crop = array[:,:-abs(tx),:]
        new_array[:,tx:,:] = crop
        new_angle = np.clip((angle + angle * abs(tx)/320), 40, 140)
    else:
        new_angle = angle
    
    return new_array, new_angle
    
def brighten(array, min_factor=0.25, max_factor=1.75):
    """
    """

    rgb_image = Image.fromarray(array)
    hsv_image = rgb_image.convert('HSV')
    
    r = random.uniform(min_factor, max_factor)

    temporary_array = np.array(hsv_image, dtype=np.uint8)
    temporary_array[:,:,2] = np.clip(temporary_array[:,:,2] * r, 0, 255).astype(np.uint8)
    
    hsv_image = Image.fromarray(temporary_array, mode='HSV')
    rgb_image = hsv_image.convert('RGB')

    new_array = np.array(rgb_image)
    
    return new_array

