# coding: utf-8

import os
import time
import re
import numpy as np
import math
from os import listdir
from os.path import isfile, join

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

#import paramiko
#import subprocess

from numba import vectorize
from numba import jit

from scipy.ndimage import imread
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import square, disk, dilation

from PIL import Image
import pandas as pd

from utils import read_tiff, make_dir, read_images_from_directory, smooth

#---------------------------
#   Colors
#---------------------------

blue = (57 / 255.0, 106 / 255.0, 177 / 255.0)
red = (204/ 255.0, 37/ 255.0, 41/ 255.0 )
green = (62/ 255.0, 150/ 255.0, 81/ 255.0 )  
grey = (128/ 255.0, 133/ 255.0, 133/ 255.0 )
gold = (237/ 255.0, 218/ 255.0, 116/ 255.0 )

#---------------------------
#   Fonts
#---------------------------

title_font_size = 20
label_font_size = 16
ticks_font_size = 16



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

    #print('OK')


    
#----------------------------#
# Time analysis              #
#----------------------------#
def time_analysis(images, corr_images, cx, cy, rx, ry, y_pos=60, ax=None, region='all', show=False, color='k'):
    
    if  not ax:
        ax  = plt.subplot(111)

    #def avg_nonzero(a): 
    #    return np.mean(a[a > 0])

    all_shots_vel = []

    #for s in range(1):    
    for s in range(shots_num):
        # Get sequence for the shot
        seq = images[s*seq_length:(s+1)*seq_length]
        seq_corr = corr_images[s*seq_length:(s+1)*seq_length]

        mean_vel_values = []

        #for i in range(1):
        for i in range(seq_length): 

            im = seq[i][cy-ry:cy+ry,cx-rx:cx+rx]
            c = seq_corr[i][cy-ry:cy+ry,cx-rx:cx+rx]
            #print(cy-r_2, cx-r_2)

            # Filtering
            
            # 1. Put away large velocity outliers
            #filtered_amp = np.where(im < 10, im, 0)
            if filter_high:
                im = np.where(im < max_vel_in_pixels, im, 0)
            
                
            # 2. Drop velocities with low correlation values
            if filter_corr:
                filtered_corr = np.where(c > min_corr_value, 1, 0)
                im = im*filtered_corr
                #filtered_corr = 1.0
                
            # 3. Drop zero velocities from avareging
            if filter_zero:
                nonzero_values = im[im > min_vel_in_pixels]
                if len(nonzero_values) == 0:
                    im = 0.0
                else:
                    im = np.mean(nonzero_values)
            
                
            
            #amp_nonzeros_mean = np.mean(filtered_amp[filtered_amp > 0])
            if not (filter_high and filter_zero):
                im = np.mean(im)
                

            # Convert to m/s
            im = im*vel_factor
            #amp_nonzeros_mean = amp_nonzeros_mean*vel_factor

            mean_vel_values.append(im)
            #mean_vel_values.append(amp_nonzeros_mean)

        all_shots_vel.append(mean_vel_values)
        #ax.plot(mean_vel_values, linewidth=1.0, alpha=0.3)

    smooth_mask = 4
    all_shots_mean = np.mean(np.array(all_shots_vel), axis=0)
    ax.plot(smooth(all_shots_mean, smooth_mask, 'flat'), linewidth=2.0, color=color, linestyle='dashed')
    
    ax.set_title('Velocity evolution, region: ' + region, size=title_font_size)
    
    x_lables = ['{:.1f}'.format(x) for x in np.arange(0, (0+seq_length)+1, 5)*sample_rate / 1000.0]
    x_ticks = np.arange(0, seq_length+1, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_lables, fontsize = ticks_font_size)
    
    ax.set_ylabel('velocity, [m\s]', size=label_font_size)
    ax.set_xlabel('time from injection, [s]', size=label_font_size)


    overall_mean = np.mean(all_shots_mean)
    overall_std = np.std(all_shots_mean)

    #print('Overall average in region {0:.2f} m/s'.format(overall_mean))
    #print('Overall variation in region {0:.2f} m/s'.format(overall_std))
    #print('Variation in percent {0:.1f} %'.format(overall_std / overall_mean * 100))
    
    ax.text(1, y_pos, 'Mean: {0:.2f} m/s'.format(overall_mean), fontsize=14, color=color)
    ax.text(26, y_pos, 'STD: {0:.2f} m/s'.format(overall_std), fontsize=14, color=color)

    ax.set_ylim(vel_min,vel_max)
    ax.yaxis.set_tick_params(labelsize=label_font_size)
    
    if show:
        plt.show()
        
    return overall_mean, overall_std


def make_analysis_figure():

    #------------------------------
    # Start analysis
    #------------------------------

    print('Analysis...')
    
    width = images.shape[2]
    height = images.shape[1]
    
    print('Image size', width, height)
    
    cx = int(width / 2) 
    cy = int(height / 2)
    
    rx = 10
    ry = 10

    # Make masking
    #med = np.median(images, axis=0)  
    #thres = np.where(med > 5, 255.0, 0.0)
    #dilated = dilation(thres, disk(6))
    #mask = gaussian_filter(dilated, 3.5) 
    #im_res = Image.fromarray(mask)
    #im_res.save(results_path + '_mask.tif')

    # Mean velocity without any filtering
    amp_mean = np.mean(images, axis=0)
    amp_mean_unfiltered_center = np.mean(amp_mean[cy-ry:cy+ry,cx-rx:cx+rx])
    im_res = Image.fromarray(amp_mean*vel_factor)
    im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_nofilter.tif')
    
    amp_mean = np.mean(images, axis=0)
    print('Mean unfiltered, center:', amp_mean_unfiltered_center*vel_factor)
    
    #amp_median = np.median(images, axis=0)
    #print('Median unfiltered, center:', np.mean(amp_median[cy-ry:cy+ry,cx-rx:cx+rx]*vel_factor))
        
    #------------------------------------------------
    # Filtering according to physical contraints
    #------------------------------------------------
    
    filtered_corr = np.where(corr > min_corr_value, 1, 0)
      
    #amp_median = np.median(images, axis=0)
    #print('Median unfiltered:', np.mean(amp_median[cy-ry:cy+ry,cx-rx:cx+rx]*vel_factor))
    
    # Make outliers filtering
    min_vel_in_pixels = (1.0 - perc_outliers_threshold)*amp_mean_unfiltered_center
    max_vel_in_pixels = (1.0 + perc_outliers_threshold)*amp_mean_unfiltered_center
    
    print('Velocity outliers:', min_vel_in_pixels*vel_factor, max_vel_in_pixels*vel_factor)

    filtered_amp = np.where(np.all([images > min_vel_in_pixels, images < max_vel_in_pixels], axis=0), images, 0)
    amp_filtered_mean = np.apply_along_axis(avg_nonzero, 0, filtered_amp*filtered_corr)
    amp_mean_filtered_value = np.mean(amp_filtered_mean[cy-ry:cy+ry,cx-rx:cx+rx])
   
    print('Mean 50% outliers and correlation filtered:', amp_mean_filtered_value*vel_factor)
    
    min_vel_in_pixels = (1.0 - perc_filtering_threshold)*amp_mean_unfiltered_center
    max_vel_in_pixels = (1.0 + perc_filtering_threshold)*amp_mean_unfiltered_center

    filtered_amp = np.where(np.all([images > min_vel_in_pixels, images < max_vel_in_pixels], axis=0), images, 0)
    amp_filtered_mean = np.apply_along_axis(avg_nonzero, 0, filtered_amp*filtered_corr)
    im_res = Image.fromarray(amp_filtered_mean*vel_factor)
    im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_filter.tif')

    measure_name = 'Mean filtered velocity'
    measure_file = 'amp_mean_filter'

    pano = amp_filtered_mean*vel_factor

    amp_seq = images


    # Setup figure
    fig = plt.figure(1)

    fig.set_size_inches(20/2 + 2, 14, forward=True)
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.1, right=0.90, hspace=0.3 )

    gs = gridspec.GridSpec(2, 1, height_ratios=[1.3,1])

    #-------------------------------------------
    # Figure 1: Velocity maps
    #-------------------------------------------

    # Setup figure
    ax0 = plt.subplot(gs[0,:])

    imx = ax0.imshow(pano, vmin=vel_min, vmax=vel_max, cmap='inferno')
    ax0.set_title('Dataset: '+ dataset +', '+ measure_name, size=title_font_size, y=1.05)    
    ax0.set_xlabel('distance, [mm]', size=label_font_size)
    ax0.set_ylabel('distance, [mm]', size=label_font_size)

    width = pano.shape[1]
    x_lables = ['{:.1f}'.format(x) for x in np.arange(0.0, width+1, 100.0) /1000.0*pixel_size]
    
    x_ticks = np.arange(0, width+1, 100)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_lables, fontsize = ticks_font_size)

    height = pano.shape[0]

    
    y_lables = ['{:.1f}'.format(x) for x in np.arange(0.0, height+1, 100.0) /1000.0*pixel_size]
    y_ticks = np.arange(0, height+1, 100)
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_lables)

    ax0.yaxis.set_tick_params(labelsize=label_font_size)

    axins0 = inset_axes(ax0,
                       width="1.5%",  # width = 10% of parent_bbox width
                       height="100%",  # height : 50%
                       loc=3,
                       bbox_to_anchor=(1.02, 0., 1, 1),
                       bbox_transform=ax0.transAxes,
                       borderpad=0                    
                       )

    # Separation lines between combined images

    clb = plt.colorbar(imx, cax=axins0)
    clb.ax.tick_params(labelsize=ticks_font_size)

    #step = int(350 / 2)
    #step = int(250 / 2)
    
    step = int(width / 4)

    ax0.axvline(step*1,linewidth=2, color=blue, alpha=0.7)
    ax0.axvline(step*2,linewidth=2, color=red, alpha=0.7)
    ax0.axvline(step*3,linewidth=2, color=green, alpha=0.7)


    # Create a Rectangle patch
    rect1 = patches.Rectangle((step*1 - rx, height/2-ry),2*rx,2*ry,linewidth=1, linestyle='--',edgecolor=blue,facecolor='none', alpha=1.0)
    ax0.add_patch(rect1)

    rect2 = patches.Rectangle((step*2 - rx, height/2-ry),2*rx,2*ry,linewidth=1, linestyle='--',edgecolor=red,facecolor='none', alpha=1.0)
    ax0.add_patch(rect2)

    rect3 = patches.Rectangle((step*3 - rx, height/2-ry),2*rx,2*ry,linewidth=1, linestyle='--',edgecolor=green,facecolor='none', alpha=1.0)
    ax0.add_patch(rect3)

    ax0.grid(False)

    #plt.show()
    #fig.savefig(path + dataset + '_comb_' + measure[1] + '_map.png')


    #-------------------------------------------
    # Figure 2: Velocity profiles
    #-------------------------------------------

    if False:

        plt.style.use('seaborn')

        #fig.set_size_inches(25, 8, forward=True)
        #plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        data = pano

        smooth_mask = 4
        avg_width = 2

        # Subplot 2       
        vel_profile_1 = np.mean(data[:,step*1-avg_width:step*1+avg_width], axis=1)
        vel_profile_2 = np.mean(data[:,step*2-avg_width:step*2+avg_width], axis=1)
        vel_profile_3 = np.mean(data[:,step*3-avg_width:step*3+avg_width], axis=1)
        
        #print(vel_profile_A.shape)
        #print(vel_profile_A)
        #
        #print(vel_profile_1.shape)
        #print(vel_profile_1)
        #
        ##vel_profile_1 = np.mean(vel_profile_1, axis=1)
        #
        #print(vel_profile_1.shape)
        #print(vel_profile_1)
        
       # plt.plot(vel_profile_A)
       # plt.plot(vel_profile_1)
        #plt.show()
        

        np.savetxt(results_path + '_vel_profile_A.txt', vel_profile_1, fmt='%.5f')
        np.savetxt(results_path + '_vel_profile_B.txt', vel_profile_2, fmt='%.5f')
        np.savetxt(results_path + '_vel_profile_C.txt', vel_profile_3, fmt='%.5f')

        ax1 = plt.subplot(gs[1,0])
        ax1.set_title('spray region: '+region, size=title_font_size)

        ax1.plot(smooth(vel_profile_1, smooth_mask, 'flat'), linewidth=1.5, color=blue)
        ax1.plot(smooth(vel_profile_2, smooth_mask, 'flat'), linewidth=1.5, color=red)
        ax1.plot(smooth(vel_profile_3, smooth_mask, 'flat'), linewidth=1.5, color=green)
        
        ax1.set_ylim(vel_min, vel_max)
        ax1.set_ylabel('velocity, [m\s]', size=label_font_size)
        ax1.set_xlabel('distance, [mm]', size=label_font_size)
        ax1.set_xticks(y_ticks)
        ax1.set_xticklabels(y_lables, fontsize = ticks_font_size)
        ax1.yaxis.set_tick_params(labelsize=label_font_size)

        
    # New ploting rotine
    
    if plot_profiles:

        plt.style.use('seaborn')

        #fig.set_size_inches(25, 8, forward=True)
        #plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

        data = pano

        smooth_mask = 4
        avg_width = 2
         
        #num_points = 7
        #
        #if region == '0': 
        #    offset = 300 # Manuall offset for the Tile 0
        #else:
        #    image_width = width*pixel_size / 1000 # in mm
        #    shift = 2.5 # in mm
        #    offset = int((image_width - shift) * 1000 / pixel_size) # in pixels
        #    offset = 50
#
        #print('Offset:', offset)
        #
        ## Profiles along spraying direction
        #for i in range(0, int((width-offset)/step_in_pixels)):
        #    
        #    x = int(offset + step_in_pixels*i)
        #    
        #    ax0.axvline(x,linewidth=1, color=grey, alpha=1.0)
        #    
        #    vels = []
        #    
        #    for p in range(num_points):
        #        
        #        y = int(height / (num_points + 1) * (p+1))
        #        
        #        rect = patches.Rectangle((x - px, y-py),2*px,2*py,linewidth=1,edgecolor=blue,facecolor='none', alpha=1.0)
        #        ax0.add_patch(rect)
        #        
        #        vel_patch_mean =  np.mean(data[y-py:y+py, x-px:x+px])
        #        vels.append(vel_patch_mean)
        #    
        #    #print(vels)
      
        
        # Subplot 2  
        
        vel_profile_1 = np.mean(data[:,step*1-avg_width:step*1+avg_width], axis=1)
        vel_profile_2 = np.mean(data[:,step*2-avg_width:step*2+avg_width], axis=1)
        vel_profile_3 = np.mean(data[:,step*3-avg_width:step*3+avg_width], axis=1)
        


        np.savetxt(results_path + '_vel_profile_A.txt', vel_profile_1, fmt='%.5f')
        np.savetxt(results_path + '_vel_profile_B.txt', vel_profile_2, fmt='%.5f')
        np.savetxt(results_path + '_vel_profile_C.txt', vel_profile_3, fmt='%.5f')

        ax1 = plt.subplot(gs[1,0])
        ax1.set_title('spray region: '+region, size=title_font_size)

        ax1.plot(smooth(vel_profile_1, smooth_mask, 'flat'), linewidth=1.5, color=blue)
        ax1.plot(smooth(vel_profile_2, smooth_mask, 'flat'), linewidth=1.5, color=red)
        ax1.plot(smooth(vel_profile_3, smooth_mask, 'flat'), linewidth=1.5, color=green)
        
        ax1.set_ylim(vel_min, vel_max)
        ax1.set_ylabel('velocity, [m\s]', size=label_font_size)
        ax1.set_xlabel('distance, [mm]', size=label_font_size)
        ax1.set_xticks(y_ticks)
        ax1.set_xticklabels(y_lables, fontsize = ticks_font_size)
        ax1.yaxis.set_tick_params(labelsize=label_font_size)
        
    
    #-------------------------------------------
    # Figure 3: Velocity time evolution
    #-------------------------------------------
    
    if plot_time_evolution:
        # Subplot 2
        ax4 = plt.subplot(gs[2,0])
        avg_vel_A, std_vel_A = time_analysis(amp_seq, corr, int(width/4), int(height/2), rx, ry, vel_min +25, ax4, region=region, show=False, color=blue)
        avg_vel_B, std_vel_B = time_analysis(amp_seq, corr, int(width/2), int(height/2), rx, ry, vel_min +15, ax4, region=region, show=False, color=red)
        avg_vel_C, std_vel_C = time_analysis(amp_seq, corr, int(3*width/4), int(height/2), rx, ry, vel_min +5, ax4, region=region, show=False, color=green)


    #all_results_path = 'y:\\projects\\pn-reduction\\2018_03_esrf_mi1325\\Phantom\\Glasduese\\analysis_all_results\\'

    fig.savefig(results_path + dataset + '_comb_'+ measure_file +'_fig.png')
    #fig.savefig(all_results_path + region + '\\' + d + '_' + dataset + '_' + region + '_comb_'+ measure_file +'_fig.png')


    #-------------------------------------------
    # Summary results table
    #-------------------------------------------

    #res = [date, dataset, region, avg_vel_A, std_vel_A, avg_vel_B, std_vel_B, avg_vel_C, std_vel_C]

    #df.loc[len(df)] = res

    print('OK')
    #results_table.append(res)

    if show_plot:
        plt.draw()
        plt.show()
    
    return amp_seq, corr, amp_filtered_mean*vel_factor
          
def count_zero(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a < min_vel_in_pixels, 1, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.sum(vals)
   
def max_outliers(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a > max_vel_in_pixels, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.max(vals)

def high_corr(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a > min_corr_value, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.mean(vals)
    
def low_corr(a):  
    eps = 1e-5
    #vals = a[a < eps]
    vals = np.where(a < min_corr_value, a, 0)
    
    if len(vals) == 0:
        return 0.0
    else:
        return np.mean(vals)

    
def plot_metrics():
    
    #a = np.array([[0,0,1,3,0], [0,0,1,3,1]])
    corr_max = np.apply_along_axis(high_corr, 0, corr)
    corr_min = np.apply_along_axis(low_corr, 0, corr)

    print(np.mean(corr_min))

    amp_zeros = np.apply_along_axis(count_zero, 0, amp)
    amp_outliers = np.apply_along_axis(max_outliers, 0, amp)
    #print(amp_zeros)

    fig = plt.figure(1)

    fig.set_size_inches(20/3 + 2, 14, forward=True)
    plt.subplots_adjust(top=0.95, bottom=0.06, left=0.1, right=0.90, hspace=0.3 )

    gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,1,1])


    # Setup figure
    ax0 = plt.subplot(gs[0,:])
    imx0 = ax0.imshow(corr_max, cmap='inferno')
    ax0.set_title('Keep correlation, Mean')
    ax0.grid(False)
    plt.colorbar(imx0, ax=ax0)

    ax1 = plt.subplot(gs[1,:])
    imx1 = ax1.imshow(corr_min, cmap='inferno')
    ax1.set_title('Drop correlation, Mean')
    ax1.grid(False)
    plt.colorbar(imx1, ax=ax1)

    ax2 = plt.subplot(gs[2,:])
    imx2 = ax2.imshow(amp_zeros, cmap='inferno')
    ax2.set_title('Low velocity, count')
    ax2.grid(False)
    plt.colorbar(imx2, ax=ax2)

    ax3 = plt.subplot(gs[3,:])
    imx3 = ax3.imshow(amp_outliers, cmap='inferno')
    ax3.set_title('Fast outliers velocity, Max')
    ax3.grid(False)
    plt.colorbar(imx3, ax=ax3)

    plt.show()

    fig.savefig(results_path + dataset + '_metrics_fig.png')
    
def check_data_frames():
    
    fig = plt.figure(1)

    fig.set_size_inches(14, 7, forward=True)
    plt.subplots_adjust(top=0.95, bottom=0.04, left=0.05, right=0.95, hspace=0.3 )

    gs = gridspec.GridSpec(2, 2)
    
    flat1 = images[flats_start]
    ax1 = plt.subplot(gs[0,0])
    ax1.imshow(flat1, cmap='gray')
    ax1.set_title('First flat field. index={}'.format(flats_start))
    plt.grid(False)
    

    flat2 = images[flats_end]
    ax2 = plt.subplot(gs[0,1])
    ax2.imshow(flat2, cmap='gray')
    ax2.set_title('Last flat field. index={}'.format(flats_end))
    plt.grid(False)


    im1 = images[start_spray]
    ax3 = plt.subplot(gs[1,0])
    ax3.imshow(im1, cmap='gray')
    ax3.set_title('First data frame. index={}'.format(start_spray))
    plt.grid(False)
 

    im2 = images[start_spray+seq_length]
    ax4 = plt.subplot(gs[1,1])
    ax4.imshow(im2, cmap='gray')
    plt.grid(False)
    ax4.set_title('Last data frame. index={}'.format(start_spray))
    
    plt.show()
    
    im1 = images[start_spray][y0:y0+h, x0:x0+w]
    plt.grid(False)
    plt.imshow(im1, vmin=30, vmax=150, cmap='gray')
    plt.title('Cropped frame. Input for processing')
    
    plt.show()
    

def summarize_velocity_profiles(data, df, pos, num_vert_point=5):
    
    width = data.shape[1]
    height = data.shape[0]

    if region == '0': 
        offset = 306 # Manuall offset for the Tile 0
    else:
        image_width = width*pixel_size / 1000 # in mm
        shift = 2.5 # in mm
        offset = int((image_width - shift) * 1000 / pixel_size) + int(step_in_pixels / 2)  # in pixels
        
    #print('Offset:', offset)

    # Profiles along spraying direction
    
    num_of_profiles = int((width-offset)/step_in_pixels)
    
    for i in range(0, num_of_profiles+1):

        x = int(offset + step_in_pixels*i)
        
        #ax.axvline(x,linewidth=1, color=grey, alpha=1.0)

        vels = []

        for p in range(num_vert_point):

            y = int(height / (num_vert_point + 1) * (p+1))
            
            #rect = patches.Rectangle((x - px, y-py),2*px,2*py,linewidth=1,edgecolor=blue,facecolor='none', alpha=1.0)
            #ax.add_patch(rect)
            
            vel_patch_mean =  np.mean(data[y-py:y+py, x-px:x+px])
            vels.append(vel_patch_mean)
        
        #res = [dataset, region, pos + step_in_mm*(i+1)] + ['{0:.1f}'.format(x) for x in vels]
        res = [dataset, region, pos + step_in_mm*(i+1)] + vels
        
        df.loc[len(df)] = res
        
    
    return pos + step_in_mm*(i+1)   


def avg_nonzero(a): 
    
    non_zero_vals = a[a > 0]
    
    if len(non_zero_vals) == 0:
        return 0.0
    else:
        return np.mean(non_zero_vals)
    
def std_nonzero(a): 
    
    non_zero_vals = a[a > 0]
    
    if len(non_zero_vals) == 0:
        return 0.0
    else:
        return np.std(non_zero_vals)
    
@jit
def avg_non_zero_numba(a):
    num = 0
    sum_val = 0
    
    for i in range(len(a)):
        if a[i] > 0:
            sum_val += a[i]
            num += 1
    if num == 0:
        return 0
        
    return sum_val / num

@jit
def avg_non_zero_full(a):
    
    nz = a.shape[0]
    ny = a.shape[1]
    nx = a.shape[2]
    
    res = np.zeros_like(a[0])
    
    for i in range(ny):
        for j in range(nx):
             
            num = 0
            sum_val = 0
            
            for k in range(nz):

                val = a[k,i,j]
                
                if val > 0:
                    sum_val += val                    
                    num += 1
                    
            if num == 0:
                res[i,j] = 0
            else:
                res[i,j] = sum_val / num 
                
    return res
  
def mul3(a, b, c):
    return a*b*c






#------------------------------
# Analysis settings
#------------------------------

#datasets = ['17_3_18_1', '17_3_23_1', '17_3_5_1', '17_3_7_3']
#regions = ['0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20']

regions = ['0', '2.5', '5', '7.5', '10']

dataset = '17_3_23_1'
region = '20'

path_input = 'y:\\projects\\pn-reduction\\2018_09_esrf_me1516\\Phantom\\' + dataset + '\\' + dataset + '_Tile_d' +region + '\\'
#path_input = 'y:\\projects\\pn-reduction\\2018_09_esrf_me1516\\Phantom\\test\\' + dataset + '_Tile_d' +region + '\\'
results_path = path_input
file_name = dataset + '_Tile_d' +region+'.tif'

max_read_images = 200
shots_num     = 1
seq_length    = max_read_images

sample_rate = 31.25 

pixel_size = 2.7 # pixel size in micrometers. Phantom Dataset: 2_2_3_10xNUV+tube200mm_A_
#pixel_size = 2.58 # pixel size in micrometers. Phantom. Estimated from shift

#pixel_size = 3.0 # Dataset: 17.2_2_3_10xNUV_A_
#pixel_size = 3.37 # Dataset: Schimadzu multiexposure


bunch_period = 176  # 176 ns bunch separation period
time_factor = bunch_period / 1000.0   # in microseconds
vel_factor = pixel_size / time_factor

# Filtering paramters
filter_high = True
filter_zero = True
filter_corr = True
            
#max_vel_in_pixels = 11
#min_vel_in_pixels = 5

min_corr_value = 0.2
perc_filtering_threshold = 0.3
perc_outliers_threshold = 0.5

vel_min = 100
vel_max = 170

# Radius of the pacth for overall statistics analysis
rx = 40
ry = 20

show_plot = False



start = time.time()
    
images = read_tiff(path_input + dataset + '_Tile_d' +region +'_amp_seq.tif', max_read_images)
corr   = read_tiff(path_input + dataset + '_Tile_d' +region + '_corr_seq.tif', max_read_images)
peak   = read_tiff(path_input + dataset + '_Tile_d' +region + '_peak_seq.tif', max_read_images)
flow_x   = read_tiff(path_input + dataset + '_Tile_d' +region + '_flow_x_seq.tif', max_read_images)
flow_y   = read_tiff(path_input + dataset + '_Tile_d' +region + '_flow_y_seq.tif', max_read_images)

end = time.time()
print ('Time elapsed: ', (end-start))



#------------------------------
# Processing
#------------------------------
vec_mul3 = vectorize('float64(float64, float64, float64)', target='parallel')(mul3)


width = images.shape[2]
height = images.shape[1]

cx = int(width / 2) 
cy = int(height / 2)

rx = 20
ry = 10


# 1. Mean velocity without any filtering
amp_mean = np.mean(images, axis=0)
amp_mean_unfiltered_center = np.mean(amp_mean[cy-ry:cy+ry,cx-rx:cx+rx])

im_res = Image.fromarray(amp_mean*vel_factor)
im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_nofilter.tif')
amp_mean = np.mean(images, axis=0)
print('Mean:', amp_mean_unfiltered_center*vel_factor)

# 2. Mean velocity with filtering on the correlation coefficient
filtered_corr = np.where(corr > min_corr_value, 1, 0)

filtered = images*filtered_corr
filtered_mean = avg_non_zero_full(filtered)
amp_mean_filtered_corr_center = np.mean(filtered_mean[cy-ry:cy+ry,cx-rx:cx+rx])

im_res = Image.fromarray(filtered_mean*vel_factor)
im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_filter_corr.tif')
print('corr>0.2, center:', amp_mean_filtered_corr_center*vel_factor)

# 3. Mean velocity with filtering on velocities
perc_filtering_threshold = 0.5
amp_mean_truth = 10

min_vel_in_pixels = (1.0 - perc_filtering_threshold)*amp_mean_truth
max_vel_in_pixels = (1.0 + perc_filtering_threshold)*amp_mean_truth

filtered_amp_seq = np.where(np.all([images > min_vel_in_pixels, images < max_vel_in_pixels, np.abs(flow_x) > np.abs(flow_y)], axis=0), images, 0)

filtered_amp = avg_non_zero_full(filtered_amp_seq)
amp_mean_filtered_center = np.mean(filtered_amp[cy-ry:cy+ry,cx-rx:cx+rx])

im_res = Image.fromarray(filtered_amp*vel_factor)
im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_filter_amp.tif')
print('amp +/- 50%:', amp_mean_filtered_center*vel_factor)

# 4. Mean velocity with filtering on local peak height
filtered_peak = np.where(peak > corr / 2.0, 1, 0)

filtered_amp = avg_non_zero_full(vec_mul3(images,filtered_corr,filtered_peak))
amp_mean_filtered_center = np.mean(filtered_amp[cy-ry:cy+ry,cx-rx:cx+rx])

im_res = Image.fromarray(filtered_amp*vel_factor)
im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_filter_peak.tif')
print('peak:', amp_mean_filtered_center*vel_factor)

# 5. Mean velocity with filtering on all constraints
filtered_amp = avg_non_zero_full(vec_mul3(filtered_amp_seq,filtered_corr,filtered_peak))
amp_mean_filtered_center = np.mean(filtered_amp[cy-ry:cy+ry,cx-rx:cx+rx])

im_res = Image.fromarray(filtered_amp*vel_factor)
im_res.save(results_path + dataset + '_Tile_d' +region + '_amp_mean_filter_all.tif')
print('all:', amp_mean_filtered_center*vel_factor)

print('Finished!')