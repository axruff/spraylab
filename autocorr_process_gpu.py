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


exec_path     = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr-concert/'
corr_exec     = 'autocorr'

#path_flow_input = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr/data/'
input_frame_file = 'frame_corr.raw'

#path_proc = 'proc/'
#path_temp = path_proc +'temp/'

clean = True
use_adaptive_flats = False
flipped = False

# Region of interest
x0 = 0          
y0 = 0             
w  = 1024
h  = 512


w  = 251
h  = 251

def process_frame_gpu(frame_n):

    # select frame
    im = images[frame_n]

    # Flat correction
    if use_adaptive_flats: 
        flat = get_similar_flat(im, sigma)
    else:
        flat = np.mean(images[1:], axis=0)

    #im = np.log((flat.astype(float)  + 0.001) / (im.astype(float)  + 0.001))
    im = np.log((1.0  + 0.001) / (im.astype(float)  + 0.001))

    # Crop frame
    im = im[y0:y0+h, x0:x0+w]
    #flat = flat[y0:y0+h, x0:x0+w]
    
    im_res = Image.fromarray(im)
    im_res.save(output_path + str(frame_n).zfill(4) + 'corr-flat.tif')

    # Flip, so the motion is to the right
    if flipped:
        im = np.fliplr(im)
        #flat = np.fliplr(flat)

    # Save input for procesing script (CUDA flow or corr)
    im_res = Image.fromarray(im)
    im.astype('float32').tofile(path_flow_input + input_frame_file)

    # Run processing for a single frame  

    command = [exec_path + corr_exec, path_flow_input + input_frame_file , str(w), str(h), output_path, str(frame_n).zfill(4), str(gpu_num)]
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

    
def read_files_save_as_multitiff_stack(path, file_name, mask=""):
    if mask == "":
        files = sorted([f for f in listdir(path) if isfile(join(path, f))])
    else:
        files = sorted([f for f in listdir(path) if isfile(join(path, f)) and f.find(mask) != -1])
    
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
    

def save_seq_as_multitiff_stack(images, file_name):
    imlist = []
    for i in range(len(images)):

        imlist.append(Image.fromarray(images[i]))
        #im.fp.close()


    imlist[0].save(file_name, save_all=True, append_images=imlist[1:])
    

#----------------------------------------
# Make dataset list  
#----------------------------------------

# Experiment: 2018_09_ersf_mi1516
#datasets = ['17_3_18_1', '17_3_23_1', '17_3_5_1', '17_3_7_3']

#datasets = ['17.1_1_1', '17.2_2_3', '17_2_10_5', '17_2_9_3']
datasets = ['17_2_10_5', '17_2_9_3']

regions = ['0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20']
start_events_offsets = np.array([0,2,4,6,8, 10,12,14,16])


# Experiment: 2018_03_ersf_mi1315
#datasets = ['007_1']
#datasets = ['018_1']
#regions = ['Z0Y0', 'Z2.5Y0', 'Z5Y0']
#start_events_offsets = np.array([0,2,4])


# Testing parameters scan
datasets = ['params_scan']
#regions = ['ei', 'mbs', 'md', 'mmi', 'mpi']
regions = ['all']


#start_events_offsets = np.array([16])

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
for dt in datasets:

    # For all regions
    for r in regions:

        dataset = dt
        region = r
        file_name = dataset + '_Tile_d' +region+'.tif'
        #file_name = dataset + '_d' +region+'.tif'
        dataset_path = u'/mnt/LSDF/projects/pn-reduction/2018_09_esrf_me1516/Phantom/' + dataset + '/'
        
        #file_name = 'OP_1bar_25C_100bar_25C.tif'
        #dataset_path = (u'/mnt/LSDF/projects/pn-reduction/2018_03_esrf_mi1325/Phantom/Glasduese/Nachtschicht 10.3 auf 11.3/'
        #                + dataset +
        #'/OP_1bar_25°C_100bar_25°C/' + 
        #                region + '/')
        
        # Experiments: Params scan
        dataset_path = u'/mnt/LSDF/projects/pn-reduction/ershov/' + dataset + '/'
        file_name = dataset + 'watershed-param-scan-' +region+'.tif'    
        
        print('\n')
        print('Dataset:', dataset)
        print('Region:', region)
        
        print(dataset_path)
        #print('Data path:', dataset_path)
        #print('Data file name:', file_name)

        #path_proc = dataset + '_d' +region + '/'
        #path_proc = dataset + '_Tile_d' +region + '/'
        path_proc = dataset + '_param_' +region + '/'
        path_temp = path_proc +'temp/'


        #max_read_images = 3348 # 2018_09_ersf_mi1516
        #max_read_images = 2882 # 2018_03_ersf_mi1315
        
        max_read_images = 23
        

        # Read dataset
        print('Reading multi-tiff file', max_read_images, 'images')

        start = time.time()


        images = read_tiff(dataset_path + file_name, max_read_images)

        end = time.time()
        

        print('Finished reading')
        print('Time elapsed: ', (end-start))

        output_path = dataset_path + path_temp
        path_flow_input = output_path

        make_dir(output_path)

        # Experiment: 2018_09_ersf_mi1516
        #spray_duration = 94
        #spray_events_separation = 224
        
        # Experiment: 2018_03_ersf_mi1315
        #spray_duration = 90
        #spray_events_separation = 280
        
        
        current_offset = start_events_offsets[regions.index(r)]
        start_indexes = np.arange(66,3300, spray_events_separation) + current_offset # Experiment: 2018_09_ersf_mi1516
        #start_indexes = np.arange(23,2800, spray_events_separation) + current_offset # Experiment: 2018_03_ersf_mi1315
        
        end_indexes   = start_indexes + spray_duration
        shot_events = range(14) # Experiment: 2018_09_ersf_mi1516
        #shot_events = range(10) # Experiment: 2018_03_ersf_mi1315
 


        #-------------------------------------
        # Adaptive flat field correction
        #-------------------------------------

        use_adaptive_flats = False

        if use_adaptive_flats:
            # Get all flats
            sigma = 15          # sigma for low-pass filtering
            
            # Experiment: 2018_09_ersf_mi1516
            flat_num = 20       # number of flats prior to each shot
            flat_offset = 20       # offset from the start of spraying
            
            # Experiment: 2018_03_ersf_mi1315
            #flat_num = 10       # number of flats prior to each shot
            #flat_offset = 10       # offset from the start of spraying

            flats = []
            flats_low_pass = []

            # For all shots
            for i in range(len(shot_events)):
                # Extract flats (images before start index)
                for k in range(start_indexes[i]-flat_num-flat_offset, start_indexes[i]-flat_offset):
                    flats.append(images[k])
                    flats_low_pass.append(gaussian_filter(images[k], sigma=sigma))

        #print('Total flats: ', len(flats))
        #print('Number of shots: ', len(shot_events))

        #save_seq_as_multitiff_stack(flats, dataset_path + path_proc + 'all_flats.tif') 


        # Make frames list            
        frames = []
        batch_size = 50 #40
        #batch_size = 50 #40
        every_nth = 1

        for i in shot_events:
            start = start_indexes[i]
            end = end_indexes[i]
            c = int(start + (end - start) / 2)
            frames.extend(list(range(c-int(batch_size/2), c+int(batch_size/2), every_nth)))


        frames = range(23)   
        print('Frames:', frames)

        selected_images = images[frames]
        #save_seq_as_multitiff_stack(selected_images, dataset_path + path_proc + 'all_input.tif') 
        
        #print('OK. Next...')
        #continue

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
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_amp_seq.tif', (h,w), 'corr-amp')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_corr_seq.tif', (h,w), 'corr-coeff')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_flow_x_seq.tif', (h,w), 'corr-flow-x')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_flow_y_seq.tif', (h,w), 'corr-flow-y')
        read_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_flat_seq.tif', 'corr-flat')
        read_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_res_seq.tif', 'corr-res')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset +'_Tile_d' +region+'_peak_seq.tif', (h,w), 'corr-peak-h')

        # Clean output folder
        if clean:
            #os.system('rm ' + output_path +'*')
            os.system('rm -r ' + output_path)


        print('OK')
