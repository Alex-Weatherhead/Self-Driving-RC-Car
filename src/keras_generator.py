import numpy as np
from PIL import Image
from keras.utils import Sequence

class KerasGenerator(Sequence):
    """
    """

    def __init__(self, items, batch_size, input_shape, preprocessor, augmenter=None):
        """
        
        Args:
            items -- a list of tuples, each containing (filepath, label).
            batch_size -- the batch_size of the Sequence.
            input_shape -- the shape of the image data.
            preprocessor -- a function which, given an image, performs the appropriate preprocessing on that image, and returns it.
        """
        self.items = items
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.preprocessor = preprocessor
        self.augmenter = augmenter
    
    def __len__(self):

        return int(np.ceil(len(self.items) / self.batch_size))
    
    def __getitem__(self, index):

        items_in_batch = self.items[(index * self.batch_size):((index + 1) * self.batch_size)]
        
        X = np.empty((self.batch_size,) + self.input_shape, dtype=np.float32)
        y = np.empty((self.batch_size), dtype=np.float32)
        
        for i, item_in_batch in enumerate(items_in_batch):
            
            label, filepath = item_in_batch
            
            image = np.array(Image.open(filepath), dtype=np.uint8)
            
            if self.augmenter:
                image, label = self.augmenter(image, label)
            
            X[i] = self.preprocessor(image)
            y[i] = -(((label - 40) / (140 - 40)) * (2) - 1)

        #print(y)
        return X, y