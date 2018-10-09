# coding: utf-8

import numpy as np
import sys
import math
import os
import time
import pickle

from os import listdir
from os.path import isfile, join

import subprocess

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage import imread
from scipy.signal import fftconvolve
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

import PIL
from PIL import Image

import multiprocessing as mp

from tqdm import tqdm

from correlation import compute_flow_area, get_spraying_events
from utils import read_tiff, make_dir


exec_path     = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr/'
corr_exec     = 'autocorr'

path_flow_input = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr/data/'
input_frame_file = 'frame_corr.raw'

path_proc = 'proc/'
path_temp = path_proc +'temp/'

clean = True
use_adaptive_flats = True
flipped = False

# Region of interest
x0 = 0          
y0 = 0             
w  = 1024
h  = 512


def process_frame_gpu(frame_n):

    # select frame
    im = images[frame_n]

    # Flat correction
    if use_adaptive_flats: 
        flat = get_similar_flat(im, sigma)
    else:
        flat = np.mean(images[1:], axis=0)

    im = np.log((flat.astype(float)  + 0.001) / (im.astype(float)  + 0.001))

    # Crop frame
    im = im[y0:y0+h, x0:x0+w]
    #flat = flat[y0:y0+h, x0:x0+w]

    # Flip, so the motion is to the right
    if flipped:
        im = np.fliplr(im)
        #flat = np.fliplr(flat)

    # Save input for procesing script (CUDA flow or corr)
    im_res = Image.fromarray(im)
    im.astype('float32').tofile(path_flow_input + input_frame_file)

    # Run processing for a single frame  

    command = [exec_path + corr_exec, path_flow_input + input_frame_file , str(w), str(h), output_path, str(frame_n).zfill(3), str(gpu_num)]
    subprocess.check_output(command)
    #subprocess.call(command)

    #print('Frame {0} '.format(i))


def get_similar_flat(image, sigma):
    diff_values = []
    
    image_low_pass = gaussian_filter(image, sigma=sigma)
    
    ixgrid = np.ix_(range(0,image.shape[0],100),range(0,image.shape[1],100))
    
    for fl in flats_low_pass:
        diff_values.append(np.mean(np.abs(image_low_pass[ixgrid] - fl[ixgrid])))
        
    min_index = np.argmin(diff_values)
    
    return flats[min_index]

def read_flow_from_components(file_u, file_v, shape):
    u = np.fromfile(file_u, dtype='float32', sep="")
    u = u.reshape(shape)
    
    v = np.fromfile(file_v, dtype='float32', sep="")
    v = v.reshape(shape)
    
    return u,v

def read_raw_image(file_name, shape):
    img = np.fromfile(file_name, dtype='float32', sep="")
    img = img.reshape(shape)
    
    return img
    
def read_raw_files_save_as_multitiff_stack(path, file_name, shape, mask=""):
    if mask == "":
        files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    else:
        files = sorted([f for f in listdir(path) if isfile(join(path, f)) and f.find(mask) != -1])
        
    #print('Number of images to convert:', len(files))
    
    imlist = []
    for f in files:
        #im = np.array(Image.open(p + f))
        #imlist.append(Image.fromarray(m))

        np_im = read_raw_image(path + f, shape)
        imlist.append(Image.fromarray(np_im))

    imlist[0].save(file_name, save_all=True, append_images=imlist[1:])

    
def read_files_save_as_multitiff_stack(path, file_name):
    files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    print(len(files))
    
    #print(PIL.version.__version__)

    imlist = []
    for f in files:
        #im = np.array(Image.open(p + f))
        #imlist.append(Image.fromarray(m))
        #print(path + f)
        #with Image.open(path + f) as im:
        #    np_im = np.array(im)
        #    imlist.append(Image.fromarray(np_im))
            
        im = Image.open(path + f, mode='r')
        np_im = np.array(im)
        imlist.append(Image.fromarray(np_im))
        #im.fp.close()


    imlist[0].save(file_name, save_all=True, append_images=imlist[1:])

    print('Convertion to multitiff is done')
    
    
    

print('\n')
print('--------------------------------------------')
print(' Correlation Processing')
print('--------------------------------------------')


if len(sys.argv) > 1:
    gpu_num = sys.argv[1]
else:
    gpu_num = 0
    

print('GPU', gpu_num)

proc_count = 0
#----------------------------------------
# Select dataset for processing
#----------------------------------------
#for dt in dataset_list[0:2]:
#for dt in dataset_list:

for x in range(1):
    
    dataset = '17_3_7_3'
    region = '0'
    file_name = dataset + '_Tile_d' +region+'_short.tif'
    dataset_path = u'/mnt/LSDF/projects/pn-reduction/2018_09_esrf_me1516/Phantom/17_3_7_3/'
    

    print('\n')
    print('Dataset:', dataset)
    print('Region:', region)
    print('Data path:', dataset_path)
    print('Data file name:', file_name)
    

    max_read_images = 50

    # Read dataset
    print('Reading multi-tiff file', max_read_images, 'images')

    start = time.time()


    images = read_tiff(dataset_path + file_name, max_read_images)

    end = time.time()

    print('Finished reading')
    print('Time elapsed: ', (end-start))
    
    output_path = dataset_path + path_temp
    
    make_dir(output_path)
        
        
    #Simulation
    start_indexes = [42] 
    end_indexes   = np.array(start_indexes) + 80
    shot_events = [0]
    
    
    #-------------------------------------
    # Adaptive flat field correction
    #-------------------------------------

    use_adaptive_flats = True

    if use_adaptive_flats:
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
                
                
    # Make frames list            
    frames = []
    batch_size = 70 #40
    every_nth = 1

    #for i in shot_events:
    #    start = start_indexes[i]
    #    end = end_indexes[i]
    #    c = int(start + (end - start) / 2)
    #    frames.extend(list(range(c-int(batch_size/2), c+int(batch_size/2), every_nth)))
    #    
        
    frames = [43,44,45]    
    print(frames)
    
    
    print('\n')
    print('Start computations of', len(frames), 'frames')
    start = time.time()
    
    
    for i in tqdm(frames):
        process_frame_gpu(i)
        
    
    end = time.time()
    print('Computation is finished')
    print ('Time elapsed: ', (end-start))
    
    print('\n')
    print('Collecting results')
    
    # Collect results
    read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_amp_seq.tif', (h,w), 'corr-amp')
    read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_corr_seq.tif', (h,w), 'corr-coeff')
    read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_flow_x_seq.tif', (h,w), 'corr-flow-x')
    read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_flow_y_seq.tif', (h,w), 'corr-flow-y')
    
    
    # Clean output folder
    if clean:
        #os.system('rm ' + output_path +'*')
        os.system('rm -r ' + output_path)
        
    
    print('OK')