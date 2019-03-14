import numpy as np
import math
import os
import re
from time import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image





def read_tiff(path, n_images):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
 
    img = Image.open(path)
    #images = np.array()
    images = []
    for i in range(n_images):
        try:
            img.seek(i)

            #np.stack(images, np.array(img))
            images.append(np.array(img))

        except EOFError:
            # Not enough frames in img
            break

    return np.array(images)

def read_images_from_directory(path):
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    print('Number of images in directory:', len(files))

    imlist = []
    for f in files:
        #im = np.array(Image.open(p + f))
        #imlist.append(Image.fromarray(m))
        with Image.open(path + f) as im:
            np_im = np.array(im)
            imlist.append(np_im)

    return np.array(imlist)

def scan_directory_tree(rootDir):
    
    path_list = []
    
    print('Start scanning')
    # Set the directory you want to start from
    #rootDir = u'y:\\projects\\pn-reduction\\2018_03_esrf_mi1325\\Phantom\\Glasduese\\'
    for dirName, subdirList, fileList in os.walk(rootDir):
        
        final_dir = (dirName.split('\\')[-1:][0])
        
        # Find Z**Y** pattern as final folders containing .cine files
        is_valid = re.search('^([Z][0-9.]+[Y][0-9.]+)', final_dir)
        #is_valid = True
         
        if is_valid:
            #path_list.append(dirName)
            #print('Dir: %s' % dirName)
            
            for fname in fileList:
                if fname.find('.cine') != -1:
                    #print('\t%s' % fname)
                    path = os.path.join(dirName, fname)
                    path_list.append(path)
                    #print('%s' % path)

    print ('End scanning')
    print ('In total', len(path_list), 'datasets')
    return path_list
    
def update_slice(images_list, sliceN):
    plt.imshow(images_list[sliceN], cmap=cm.gray)
    #plt.colorbar()
    plt.show()

def make_dir(path):

    try:  
        os.makedirs(path)
        
    except OSError:
        pass
        #print ("Creation of the directory %s failed" % path)    
        
