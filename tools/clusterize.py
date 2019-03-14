import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from vtk import vtkStructuredPointsReader
from vtk.util import numpy_support as VN
from PIL import Image
from scipy.ndimage import imread
from tqdm import tqdm


def save_3d_array_as_multitiff(array, file_name):

    imlist = list(array)
    imlist[0].save(file_name, save_all=True, append_images=imlist[1:])
    
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def read_images_from_directory(path):
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    print('Number of images in directory:', len(files))

    imlist = []
    for f in files:
        with Image.open(path + f) as im:
            np_im = np.array(im)
            imlist.append(np_im)

    return np.array(imlist)


def save_np_sequence_as_multitiff_stack(images, file_name):
    
    imlist = []
    for i in range(len(images)):
        imlist.append(Image.fromarray(images[i]))

    imlist[0].save(file_name, save_all=True, append_images=imlist[1:])
    
    del imlist
    
    
    
input_path = 'y:\\projects\\pn-reduction\\simulation\\dt51ns_17timesteps\\'

vol = read_images_from_directory(input_path + '\\01_den\\')

frame = 145

plt.imshow(vol[frame], cmap='gray')
plt.show()

from scipy.ndimage import sobel, generic_gradient_magnitude

# Compute gradients
grad = generic_gradient_magnitude(vol, sobel)

plt.imshow(grad[frame], cmap='gray')
plt.show()

print(np.max(grad))

# Find esdes via thresholding
edges = np.where(grad > 2000, 0, 1)

plt.imshow((vol*edges)[frame], cmap='gray')
plt.show()

save_np_sequence_as_multitiff_stack(vol*edges, input_path + '\\01_edges.tif')

