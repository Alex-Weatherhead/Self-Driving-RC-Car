import random
from os import listdir

def populate(dataset, directory, extension):
    """
    """
    files = listdir(directory)
    for file in files:
        if file.endswith(extension):
            angle = round(float(file[:-len(extension)].split('-')[1]))
            dataset[angle].append(directory + file)

def split(dataset, training_size=0.70, validation_size=0.30, shuffle=True):
    """
    """
    training_dataset = {}
    validation_dataset = {}

    for label in dataset:
        
        samples = dataset[label]
        number_of_samples = len(samples)
        
        if shuffle:
            random.shuffle(samples)
        
        training_dataset[label] = samples[:round(len(samples) * training_size)]
        validation_dataset[label] = samples[-round(len(samples) * validation_size):]
    
    return training_dataset, validation_dataset

def balance(dataset, downsample_threshold=75, upsample_threshold=50):
    """
    """
    for label in dataset:
    
        number_of_samples = len(dataset[label])
        
        if number_of_samples == 0:
            continue
        elif number_of_samples > downsample_threshold:
            k = downsample_threshold
            dataset[label] = random.sample(dataset[label], k=k)
        elif number_of_samples < upsample_threshold:
            k = upsample_threshold - number_of_samples
            dataset[label] += random.choices(dataset[label], k=k)

def unpack(dataset, shuffle=True):
    """
    """
    data = [(angle, file) for angle in dataset for file in dataset[angle]]
    
    if shuffle:
        random.shuffle(data)
    
    return data