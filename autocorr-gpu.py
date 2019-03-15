# coding: utf-8
import numpy as np
import sys
import math
import os
import time as t

from os import listdir
from os.path import isfile, join

from scipy.ndimage import imread
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

import subprocess
from tqdm import tqdm

from utils import *
from numerics import smooth

import data_profiles
from data_profiles import Spraying, FlatField, FileInputMode


# Processing script
exec_path     = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr-concert/'
corr_exec     = 'autocorr'
input_frame_file = 'frame_corr.raw'
#path_flow_input = '/mnt/LSDF/anka-nc-cluster/home/ws/fe0968/autocorr/data/'

# Load data processing profile
p = data_profiles.sim


#------------------------------------------
# Get parameters from a profile  
#------------------------------------------

root_path = p['root_path']
datasets = p['datasets']
params = p['params']

param_name = p['param_name']
file_input_mode = p['file_input_mode']
single_file_name = p['single_filename']

num_frames = p['num_frames']
spray_mode = p['spray_mode']
shot_events = p['spray_mode_params']['shot_events']
start_events_offsets = p['spray_mode_params']['start_events_offsets']
spray_duration = p['spray_mode_params']['spray_duration']
spray_events_separation = p['spray_mode_params']['spray_events_separation']
batch_size = p['spray_mode_params']['batch_size']
use_every_nth = p['spray_mode_params']['use_every_nth']
start_index = p['spray_mode_params']['start_index']
end_index = p['spray_mode_params']['end_index']

flat_mode = p['flat_mode']
flat_field_value = p['flat_field_value']
sigma = p['adaptive_flat']['sigma']
flat_num = p['adaptive_flat']['flat_num']
flat_offset = p['adaptive_flat']['flat_offset']

x0 = p['image_size'][0]       
y0 = p['image_size'][1]             
w  = p['image_size'][2]
h  = p['image_size'][3]
flipped = p['flat_mode']

clean_intermediate_results = p['clean_intermediate_results']



def process_frame_gpu(frame_n):

    # select frame
    im = images[frame_n]

    # Flat correction
    if flat_mode == FlatField.Adaptive:
        flat = get_similar_flat(im, sigma)
    elif flat_mode == FlatField.Mean:
        flat = np.mean(images[1:], axis=0)
    else:
        flat = flat_field_value

    if flat_mode != FlatField.No and not FlatField.Constant:
        im = np.log((flat.astype(float)  + 0.001) / (im.astype(float)  + 0.001))
    else:
        # Constant value of flat field
        im = np.log((flat  + 0.001) / (im.astype(float)  + 0.001))



    # Crop frame
    im = im[y0:y0+h, x0:x0+w]
    
    im_res = Image.fromarray(im)
    im_res.save(output_path + str(frame_n).zfill(4) + 'corr-flat.tif')

    # Flip, so the motion is to the right
    if flipped:
        im = np.fliplr(im)

    # Save input for procesing script (CUDA flow or corr)
    im_res = Image.fromarray(im)
    im.astype('float32').tofile(path_flow_input + input_frame_file)

    # Run processing for a single frame  
    command = [exec_path + corr_exec, path_flow_input + input_frame_file , str(w), str(h), output_path, str(frame_n).zfill(4), str(gpu_num)]
    subprocess.check_output(command)


def get_similar_flat(image, sigma):
    diff_values = []
    
    image_low_pass = gaussian_filter(image, sigma=sigma)
    
    ixgrid = np.ix_(range(0,image.shape[0],100),range(0,image.shape[1],100))
    
    for fl in flats_low_pass:
        diff_values.append(np.mean(np.abs(image_low_pass[ixgrid] - fl[ixgrid])))
        
    min_index = np.argmin(diff_values)
    
    return flats[min_index]

#----------------------------------------
# Make dataset list  
#----------------------------------------

print('\n')
print('------------------------------------------------------')
print(' Auotcorrelation. GPU version. Author: Alexey Ershov ')
print('-----------------------------------------------------')


if len(sys.argv) > 1:
    gpu_num = sys.argv[1]
else:
    gpu_num = 0
    

print('GPU', gpu_num)

proc_count = 0

#----------------------------------------
# Select dataset for processing
#----------------------------------------
for dt in datasets:

    # For all parameters (e.g. different regions, spray simulation parameters)
    for r in params:

        dataset = dt
        param = r

        dataset_path = root_path + dataset + '/'

        if file_input_mode == FileInputMode.Single:
            file_name = single_file_name
        elif file_input_mode == FileInputMode.DatasetNameParam:
            file_name = dataset + param_name + param +'.tif'
        elif file_input_mode == FileInputMode.DatasetParam:
            file_name = dataset + param +'.tif'
        else:
            file_name = param_name + param +'.tif'

        print('\n')
        print('Dataset:', dataset)
        print('Region:', param)
        
        print('\n')
        print('Dataset path: ', dataset_path)
        print('Dataset file name:', file_name)
        print('\n')

        path_proc = dataset + '_' + param_name + param + '/'
        path_temp = path_proc +'temp/'
   
        max_read_images = num_frames
     
        # Read dataset
        print('Reading multi-tiff file', max_read_images, 'images')

        start = t.time()
        images = read_tiff(dataset_path + file_name, max_read_images)
        end = t.time()
        
        print('Finished reading')
        print('Time elapsed: ', (end-start))

        output_path = dataset_path + path_temp
        path_flow_input = output_path

        make_dir(output_path)
 
        if spray_mode == Spraying.Multiple:
            # Compute ranges of frames with spraying data
            current_offset = start_events_offsets[params.index(r)]
            start_indexes = np.arange(start_index, end_index, spray_events_separation) + current_offset    
            end_indexes   = start_indexes + spray_duration

        #-------------------------------------
        # Adaptive flat field correction
        #-------------------------------------

        if flat_mode == FlatField.Adaptive and spray_mode == Spraying.Multiple:

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

        # Select frames for processing
         
        if spray_mode == Spraying.Single:
            frames = range(num_frames) 
        else:          
            frames = []               
            for i in shot_events:
                start = start_indexes[i]
                end = end_indexes[i]
                c = int(start + (end - start) / 2)
                frames.extend(list(range(c-int(batch_size/2), c+int(batch_size/2), use_every_nth)))
    
        #print('Frames:', frames)

        selected_images = images[frames]
        #save_seq_as_multitiff_stack(selected_images, dataset_path + path_proc + 'all_input.tif') 
        
        #print('OK. Next...')
        #continue

        print('\n')
        print('Start computations of', len(frames), 'frames')
        
        start = t.time()
        
        for i in tqdm(frames):
            process_frame_gpu(i)
        end = t.time()

        print('Computation is finished')
        print ('Time elapsed: ', (end-start))

        print('\n')
        print('Collecting results...')

        # Collect results
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset + param_name + param+'_amp_seq.tif', (h,w),      'corr-amp')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset + param_name + param+'_corr_seq.tif', (h,w),     'corr-coeff')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset + param_name + param+'_flow_x_seq.tif', (h,w),   'corr-flow-x')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset + param_name + param+'_flow_y_seq.tif', (h,w),   'corr-flow-y')
        read_raw_files_save_as_multitiff_stack(output_path, dataset_path + path_proc + dataset + param_name + param+'_peak_seq.tif', (h,w),     'corr-peak-h')
        read_files_save_as_multitiff_stack(    output_path, dataset_path + path_proc + dataset + param_name + param+'_flat_seq.tif',            'corr-flat')
        read_files_save_as_multitiff_stack(    output_path, dataset_path + path_proc + dataset + param_name + param+'_res_seq.tif',             'corr-res')

        # Clean output folder
        if clean_intermediate_results:
            print('\n')
            print('Removing intermediate result files...')
            #os.system('rm ' + output_path +'*')
            os.system('rm -r ' + output_path)
            print('OK')


        print('Done')
