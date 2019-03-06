import numpy as np
from matplotlib import pyplot

def distribution_hist(fname, dataset, yticks=None):
    
    labels = [key for key in dataset.keys() for value in dataset[key]]
    
    pyplot.hist(labels, bins=101)
    
    pyplot.xlabel('Label')
    pyplot.ylabel('Samples')
    
    if yticks:
        pyplot.yticks(yticks)

    pyplot.savefig(fname)
    pyplot.close()
    
def losses_plot(fname, training_loss, validation_loss, epochs, yticks=None):
    
    pyplot.plot(training_loss)
    pyplot.plot(validation_loss)
    
    pyplot.xlabel('Epoch')
    pyplot.ylabel('loss')
    
    pyplot.xticks(np.arange(epochs) + 1)
    if yticks:
        pyplot.xticks(yticks)
    
    pyplot.legend(['training', 'validation'], loc='upper right')
    
    pyplot.savefig(fname)
    pyplot.close()

def show_activations(feature_maps, grid=None):
    
    height, width, filters = np.shape(feature_maps)
    
    if grid:
        
        figure, axes = pyplot.subplots(*grid)
        
        for i in range(grid[0]):
            for j in range(grid[1]):
                axes[i][j].imshow(feature_maps[:,:,((i+1)*(j+1)-1)], cmap='Greys_r')
                axes[i][j].axis('off')
                axes[i][j].set_aspect("auto")
                
    else:
        
        figure, axes = pyplot.subplots(filters)
        
        for i in range(filters):
            axes[i].imshow(feature_maps[:,:,i], cmap='Greys_r')
            axes[i].axis('off')

    pyplot.subplots_adjust(hspace = 0, wspace = 0)
    pyplot.show()
    pyplot.close()
