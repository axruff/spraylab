# coding: utf-8

import numpy as np
import math
import os
import time
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage import imread
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

from PIL import Image

import multiprocessing as mp

from correlation import compute_flow_area, get_spraying_events
from utils import read_tiff, make_dir


# top directory for processing results
path_proc = 'proc_new/'

path_raw = path_proc +    'raw/'
path_input = path_proc +  'input/'
path_amp = path_proc +    'amp/'
path_flow_x = path_proc + 'flow_x/'
path_flow_y = path_proc + 'flow_y/'
path_corr = path_proc +   'corr/'
path_width = path_proc +  'width/'


def process_frame(frame_index):
    frame = frame_index
            
    #print(str(frame))
    
    # Get current image frame
    raw = images[frame]       
    im_res = Image.fromarray(raw)
    im_res.save(path + path_raw + str(frame).zfill(4) + '_orig_raw.tif')

    # Flat correction
    flat = get_similar_flat(raw, sigma)
    im = np.log((flat.astype(float)  + 0.001) / (raw.astype(float)  + 0.001))
    im_res = Image.fromarray(im)
    im_res.save(path + path_input + str(frame).zfill(4) + '_flat_corr.tif')

    #-----------------------------
    # Compute correlation
    #-----------------------------

    window_size = 30

    # Crop frame
    xmin = 250
    xmax = 950
    ymin = 50
    ymax = 450

    # Centeral patch
    #xmin = 500
    #xmax = 600
    #ymin = 180
    #ymax = 320

    # Small patch
    #xmin = 550
    #xmax = 600
    #ymin = 200
    #ymax = 250

    #plt.imshow(im[ymin:ymax, xmin:xmax], vmin=-0.5, vmax=0.5, cmap='gray')
    #plt.show()


    # Compute cropped region
    amp, vx, vy, corr, width = compute_flow_area(im, window_size, xmin, xmax, ymin, ymax)

    # Collect results
    #amp_res_list = np.stack((amp_res_list, amp), axis=0)
    #amp_res_list.append(amp)
    #corr_res_list = np.stack((corr_res_list, corr), axis=0)
    #flow_x_res_list = np.stack((flow_x_res_list, vx), axis=0)
    #flow_y_res_list = np.stack((flow_y_res_list, vy), axis=0)

    # Save results
    im_res = Image.fromarray(amp)
    im_res.save(path + path_amp + str(frame).zfill(4) +'_res_amp.tif')

    im_res = Image.fromarray(vx)
    im_res.save(path + path_flow_x + str(frame).zfill(4) +'_res_flow_x.tif')

    im_res = Image.fromarray(vy)
    im_res.save(path + path_flow_y + str(frame).zfill(4) +'_res_flow_y.tif')

    im_res = Image.fromarray(corr)
    im_res.save(path + path_corr + str(frame).zfill(4) +'_res_corr.tif')
    
    im_res = Image.fromarray(width)
    im_res.save(path + path_width + str(frame).zfill(4) +'_res_width.tif')

    
def get_similar_flat(image, sigma):
    diff_values = []
    
    image_low_pass = gaussian_filter(image, sigma=sigma)
    
    ixgrid = np.ix_(range(0,image.shape[0],100),range(0,image.shape[1],100))
    
    for fl in flats_low_pass:
        diff_values.append(np.mean(np.abs(image_low_pass[ixgrid] - fl[ixgrid])))
        
    min_index = np.argmin(diff_values)
    
    return flats[min_index]



#----------------------------------------
# Read dataset list  
#----------------------------------------

#with open ('datasets_list', 'rb') as fp:
#    datasets_path_file_list = pickle.load(fp)
    
#print(datasets_path_file_list)



debug_mode = False


print('\n')
print('--------------------------------------------')
print(' Correlation Processing')
print('--------------------------------------------')


#datasets_path_file_list = [('/Users/aleksejersov/data/spray/0_25', 'test.tif')]

#----------------------------------------
# Select dataset for processing
#----------------------------------------
#for dataset in datasets_path_file_list:
  
#path = dataset[0] + '/'
#file_name = dataset[1]

path_023 = u'/mnt/LSDF/projects/pn-reduction/2018_03_esrf_mi1325/Phantom/Glasduese/Nachtschicht 09.3 auf 10.3/023_1/Ansicht 0°/OP_1bar_25°C_100bar_25°C/Z2.5Y0/'
path_025_no_trans = u'/mnt/LSDF/projects/pn-reduction/2018_03_esrf_mi1325/Phantom/Glasduese/Nachtschicht 09.3 auf 10.3/025_1/Ansicht 0°/OP_1bar_25°C_100bar_25°C/Z2.5Y0/'
path_025 = u'/mnt/LSDF/projects/pn-reduction/2018_03_esrf_mi1325/Phantom/Glasduese/Nachtschicht 11.3 auf 12.3/025_1/Ansicht 0°/OP_1bar_25°C_100bar_25°C/Z0Y0/'

file_name = 'OP_1bar_25C_100bar_25C.tif'
             
path = path_025

print('\n')
print('Data path:', path)
print('Data file name:', file_name, '\n\n')

max_read_images = 150 #1500    

# Read dataset
print('Reading multi-tiff file', max_read_images, 'images')

start = time.time()

images = read_tiff(path + file_name, max_read_images)

end = time.time()

print('Finished reading')
print ('Time elapsed: ', (end-start))

# Make analysis directories
make_dir(path + path_raw)
make_dir(path + path_input)
make_dir(path + path_amp)
make_dir(path + path_flow_x)
make_dir(path + path_flow_y)
make_dir(path + path_corr)
make_dir(path + path_width)


# Analyze frames inside a dataset, get events starting and ending points as frame numbers
print('\nAnalyzing spraying shots')

if not debug_mode:
    start_indexes, end_indexes = get_spraying_events(images, max_read_images-1, sigma=15, min_brigthness=15, range_diff_value=0.5)
    
    # Correct start indexes
    #start_indexes = [21, 301, 582, 862, 1143, 1423, 1703, 1983, 2263, 2543, 2823] # 023_1
    #start_indexes = [22, 302, 582, 862, 1143, 1422, 1703, 1982, 2263, 2542, 2824] # 025_1
    start_indexes = [21, 301, 582, 862, 1143] # 023_1
else:
    start_indexes = [21]
    end_indexes = [21+80]

print('Done')
print(start_indexes)
print(end_indexes)
#print('Durations', np.array(end_indexes) - np.array(start_indexes))


#-------------------------------------
# Adaptive flat field correction
#-------------------------------------

# Get all flats
sigma = 15          # sigma for low-pass filtering
flat_num = 18       # number of flats prior to each shot

flats = []
flats_low_pass = []

# For all shots
for i in range(len(start_indexes)):
    # Extract flats (images before start index)
    for k in range(start_indexes[i]-flat_num, start_indexes[i]):
        flats.append(images[k])
        flats_low_pass.append(gaussian_filter(images[k], sigma=sigma))

        

# Arrays to store the integrated results
image_shape = images[0].shape
#amp_res_list = np.empty(image_shape)
amp_res_list = []
corr_res_list = np.empty(image_shape)
flow_x_res_list = np.empty(image_shape)
flow_y_res_list = np.empty(image_shape)


# DEPRICATED: Flat field correction using average over all flats
# get flats (first frame indicated by start_indexes)
#flats = images[0: start_indexes[0]-1]

# average flats
#flat = np.mean(flats, axis=0)

#im_res = Image.fromarray(flat)
#im_res.save(path + path_proc + 'flat.tif')


# Construct list of frames for processing from multiple spray shot events
if not debug_mode:
    shot_events = [0,1,2,3,4]
    #shot_events = [0]
else:
    shot_events = [0]

    
    
frames = []
batch_size = 40
every_nth = 1

for i in shot_events:
    start = start_indexes[i]
    end = end_indexes[i]
    c = int(start + (end - start) / 2)
    frames.extend(list(range(c-int(batch_size/2), c+int(batch_size/2), every_nth)))

#----------------------------------------
# Start threads for each frame
#----------------------------------------

#frames = range(25,45)

print('\n')
print('Start computations of', len(frames), 'frames')
start = time.time()


process_frame(40)


#pool = mp.Pool(processes=35)
#results = [pool.apply_async(process_frame, args=(x,)) for x in frames]
#pool.close()
#pool.join()



end = time.time()
print('Computation is finished')
print ('Time elapsed: ', (end-start))


