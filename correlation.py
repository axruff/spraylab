import numpy as np
import math
from time import time

from scipy.ndimage import label
from scipy.ndimage import imread
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops
#from skimage.filter import sobel

import matplotlib.pyplot as plt

from PIL import Image

        
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

        
def crop(im, center, window):
    return im[int(center[0]-window/2):int(center[0]+window/2), int(center[1]-window/2):int(center[1]+window/2)]


def compute_flow(corr):
    
    #corr = gaussian_filter(corr, 1.0)
    
    # indexes of local maxima peaks
    maxima_ind = peak_local_max(corr, min_distance=1, num_peaks=7)

    maxima = sorted(corr[maxima_ind[:,0], maxima_ind[:,1]])
    
    if (len(maxima) == 1):
        return 0.,0.,0.,0.,0.

    v1 = maxima[-2]
    v2 = maxima[-3]

    # select indexes of two peaks
    index1 = np.where(corr == v1)

    # when both maxima are equal
    if len(index1[0]) > 1:
        index2 = [index1[0][0], index1[1][0]] 
        index1 = [index1[0][1], index1[1][1]] 
    else:
        index2 = np.where(corr == v2)

    # movement to the right
    if index1[1] > corr.shape[1] / 2.0:
        index = index1
    else:
        index = index2

    #print index 
    
    # --------------------------------------
    # Peak width analysis
    # --------------------------------------
     # Manual markers
    #markers = np.zeros_like(corr)
    #markers[index1[0], index1[1]] = 5
    #markers[index2[0], index2[1]] = 10
    #
    ## Test 2: Watershed on edges via Sobel 
    #edges = sobel(corr-0.00001)
    #labels = watershed(edges, markers)
    #
    #max_peak_label = labels[index[0], index[1]]
    #
    #props = regionprops(labels.astype(int))
    #peak_area = 0   
    #for reg in props:
    #    if reg.label == max_peak_label:
    #        peak_area = reg.area   
            
    w = int(corr.shape[0] / 2) + 1 # patch center

    n = corr[w-1-1:w+1,w-1-1:w+1]
    avg_n = (np.sum(n) - 1.0) / 8.0
    
           
    x, y = index[1], index[0]
    vec = math.ceil(y - corr.shape[0] / 2.0), math.ceil(x - corr.shape[1] / 2.)
    corr = corr[y, x]
    
    return np.sqrt(np.dot(vec, vec)), vec[1], vec[0], corr, avg_n



def compute_flow_area(image, window, xmin, xmax, ymin, ymax, axis_to_check=1, perc=0.99, d_perc=1e-3):
    start = time()
    #print('Computing flow')
    shape = (ymax - ymin, xmax - xmin)
    dx = np.zeros(shape)
    dy = np.zeros(shape)
    amp = np.zeros(shape)
    corr = np.zeros(shape)
    peak_width = np.zeros(shape)
    errors = np.zeros(shape)
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            a = crop(image, (y, x), window)
            c = fftconvolve(a, a[::-1, ::-1], mode='full') / np.sum(a ** 2)
            
            #try:
                
            # Compute flow from the autocorrelation map
            vp, xp, yp, c, w = compute_flow(c)

            #if (vp > 15):
            #    xp = yp = vp = 0

            #if (xp < 1):
            #    xp = yp = vp = 0

            dx[y - ymin, x - xmin] = xp
            dy[y - ymin, x - xmin] = yp
            amp[y - ymin, x - xmin] = vp
            corr[y - ymin, x - xmin] = c
            peak_width[y - ymin, x - xmin] = w



           # if xp > 15:
           #     print 'Too large x at: y:', y, ' x:', x, ' Value:', xp
           #     
           # if yp < -15:
           #     print 'Negative y at: y:', y, ' x:', x, ' Value:', yp

            #except:
                #print 'Error at y:', y, ' x:', x
                #errors[y - ymin, x - xmax] = 1
        #print y
            
    #print('Finished')
    elapsed = time() - start
    #print("Time elapsed: ", elapsed)

    return amp, dx, dy, corr, peak_width


def get_spraying_events(images, n_images, sigma=15, min_brigthness=15, range_diff_value=0.4):

    w = images.shape[2]
    h = images.shape[1]

    rows = range(80, 80+100)
    cols = range(330, 330+100)
    patch_empty = np.ix_(rows, cols)


    rows = range(80, 120)
    cols = range(1000, 1010)
    patch_shutter = np.ix_(rows, cols)

    rows = range(int(h/2 - h/6), int(h/2 + h/6))
    cols = range(int(w/2 - w/6 - 50), int(w/2 + w/6 -50))
    patch_spray = np.ix_(rows, cols)

    res_range_empty = []
    res_range_spray = []
    res_mean_dark = []

    start_indexes = []
    end_indexes = []

    dark_mode = False
    spray_mode = False

    for i in range(0,n_images):

        i1 = images[i][patch_shutter].astype(float)
        if (np.mean(i1) < min_brigthness):

            if dark_mode == False:
                end_indexes.append(i)

            dark_mode = True
            continue
        else:
            if dark_mode == True: # initial conditions
                dark_mode = False
                spray_mode = False


        i1 = images[i][patch_empty].astype(float) 
        i2 = images[i+1][patch_empty].astype(float)

        f1 = gaussian_filter(i1.astype(float), sigma)
        f2 = gaussian_filter(i2.astype(float), sigma)

        diff_empty = i1 / f1 - i2 / f2

        i1 = images[i][patch_spray].astype(float)
        i2 = images[i+1][patch_spray].astype(float)

        f1 = gaussian_filter(i1.astype(float), sigma)
        f2 = gaussian_filter(i2.astype(float), sigma)

        diff_spray = i1 / f1 - i2 / f2

        range_empty = np.max(diff_empty) - np.min(diff_empty)
        range_spray = np.max(diff_spray) - np.min(diff_spray)

        res_range_empty.append(range_empty)
        res_range_spray.append(range_spray)
        res_mean_dark.append(np.mean(i1))

        if (((range_spray - range_empty) / range_spray) > range_diff_value):
            if (not dark_mode and  not spray_mode):
                spray_mode = True
                start_indexes.append(i)
            #print(i)
            #break

    if len(start_indexes) > len(end_indexes):
        end_indexes.append(n_images)
        
    return start_indexes, end_indexes



def my_correlation3(im, show=False):
    
    m = im.shape[0]
    n = im.shape[1]

    corr = np.zeros((m, n))
    
    norm_sum = 0
    
    for i in range(m):
        for j in range(n):
            norm_sum += im[i,j]**2 
    
    # for all pixels
    for i in range(-int(m/2), int(m/2)):
        for j in range(-int(n/2), int(n/2)):
            
            sum_c = 0  
            for k in range(m):
                for t in range(n):
                    if (k+i) < 0 or (k+i)>(m-1) or (t+j) < 0 or (t+j)>(n-1) :
                        continue
                    sum_c += im[k,t]*im[k+i, t+j]

            corr[int(m/2)+i+0,int(n/2)+j+0] = sum_c / norm_sum        

    if show:        
        plt.imshow(corr, cmap='jet')
        plt.colorbar()
        plt.show()
        
    return corr


def my_find_peaks(image, min_distance, peak_threshold=0.01, show=False):
    
    eps = 1e-4
    
    values = []
    
    indexes_x = []
    indexes_y = []
    

    m = image.shape[0] # y, height
    n = image.shape[1] # x, width

    # Maximum filter
    max_res = np.zeros_like(image) # not needed
    
    # for all pixels
    for i in range(m):
        for j in range(n):
            
            max_val = 0  
            for k in range(-min_distance,min_distance):
                for t in range(-min_distance, min_distance):
   
                    ind_x = j+t
                    ind_y = i+k

                    if ind_x > n-1 or ind_x < 0 or ind_y > m-1 or ind_y < 0:
                        continue

                    if image[ind_y, ind_x] > max_val:
                        max_val = image[ind_y, ind_x] 
            
            val = image[i,j] 
            if ((abs(val - max_val)) < eps) and (val > peak_threshold):
                indexes_x.append(j) # coorect 
                indexes_y.append(i)
                values.append(val)
                
            max_res[i,j] = max_val # not needed
            

    #if show:
    #    plt.imshow(max_res, cmap='jet')
    #    plt.colorbar()
    #    plt.show()   

    
    #diff = (image - max_res) # not needed
    #indexes = np.where(diff == 0) # not needed

    if show:
        plt.imshow(image, cmap='jet')
        plt.scatter(indexes_x, indexes_y)
        #plt.scatter(indexes[1], indexes[0])
        #plt.colorbar()
        plt.show()  
        
    return [indexes_y, indexes_x], values



def my_select_peak(image, indexes_y, indexes_x, values, num_peaks= 5):
    
    if (len(values) == 1):
        print('Warning: One peak!')
    
    # Selecting n largest peaks by partial sorting
    if len(values) > num_peaks:
        selectedValues = np.partition(values, len(values) -num_peaks-1)[-num_peaks:]
        #print(np.partition(values, len(values) -num_peaks-1))
    else:
        selectedValues = values
        
    maxima = sorted(selectedValues) 
    
    v1 = maxima[-2]
    v2 = maxima[-3]

    print('Peaks:', v1, v2)

    # select indexes of two peaks
    index1 = np.where(values == v1)

    #print(index1)
    
    # when both maxima are equal
    if len(index1[0]) > 1:
        index2 = index1[0][0]
        index1 = index1[0][1]
    else:
        index2 = np.where(values == v2)[0][0]

    print(index1)
    print(index2)
    
    # movement to the right
    if indexes_x[index1] > indexes_x[index2]:
        x, y = indexes_x[index1], indexes_y[index1]
    else:
        x, y = indexes_x[index2], indexes_y[index2]

    c = image[y, x]
    x -= int(image.shape[1] / 2.0)
    y -= int(image.shape[0] / 2.0)
    
    #print('Index:', index )

    #x, y = index[1], index[0]
    
    amp = np.sqrt(x**2 + y**2)
    
    print('All peaks:', values)
    print('Top', num_peaks, 'peaks:', selectedValues)
    print('Vec (y,x):', y, x)
    print('Amp:', amp)
    print('Corr:', c)
    
    return x, y, amp, c



# Compute correlation
def my_correlation(im, show=False):
    
    m = im.shape[0]
    n = im.shape[1]

    corr = np.zeros((2*m-1, 2*n-1))
    
    norm_sum = 0
    
    # for all pixels
    for i in range(m):
        for j in range(n):
            
            sum_c = 0  
            for k in range(i+1):
                for t in range(j+1):
                    sum_c += im[k,t]*im[k+(m-i)-1, t+(n-j)-1]

            corr[i,j] = sum_c 
            norm_sum += im[i,j]**2 
            
            
    # for all pixels
    for i in range(m):
        for j in range(n):
            
            sum_c = 0  
            for k in range(m-i):
                for t in range(n-j):
                    sum_c += im[k,t]*im[k+i, t+j]

            corr[m+i-1,n+j-1] = sum_c 
            
            
    # normalize correlation coefficients
    for i in range(2*m-1):
        for j in range(2*n-1):
            corr[i,j] = corr[i,j] / norm_sum
            

    if show:        
        plt.imshow(corr, cmap='jet')
        plt.colorbar()
        plt.show()
        
    return corr


# Compute correlation
def my_correlation_1d(im, show=False):

    m = im.shape[0]
    
    corr = np.zeros(2*m-1)
    
    print(corr.shape)

    norm_sum = 0
    
    # for all pixels
    for i in range(-m+1, m):       
        sum_c = 0  
        for k in range(m):

            if (k+i) < 0 or (k+i)>(m-1):
                continue
                
            sum_c += im[k]*im[k+i]

        corr[m+i-1] = sum_c 
            

    if show:        
        plt.plot(corr)
        plt.show()
        
    return corr
        