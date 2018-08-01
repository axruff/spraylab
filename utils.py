import numpy as np
import math
import os
from time import time

import matplotlib.pyplot as plt
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
        
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    #return y
    return y[(int(window_len/2)-1):-(int(window_len/2))]