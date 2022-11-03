#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:57:38 2021

@author: Mathew
"""


from skimage.io import imread
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from skimage import filters,measure
from skimage.filters import threshold_local


# Settings
image_width=512
image_height=512
Pixel_size=103.0


filename_contains="405_0"

# Folders to analyse:
root_path=r"/Users/Mathew/Documents/Current analysis/SAVE_for_Sonia/"
pathList=[]

pathList.append(r"/Users/Mathew/Documents/Current analysis/SAVE_for_Sonia/All/2uM_0haggs_PLL_2022-11-01_15-25-02/")
                
def load_image(toload):
    
    image=imread(toload)
    
    return image

def z_project(image):
    
    mean_int=np.mean(image,axis=0)
  
    return mean_int

# Subtract background:
def subtract_bg(image):
    background = threshold_local(image, 11, offset=np.percentile(image, 1), method='median')
    bg_corrected =image - background
    return bg_corrected

def threshold_image_alt(input_image):
    # threshold_value=filters.threshold_otsu(input_image)  
    
    threshold_value=input_image.mean()+5*input_image.std()
    print(threshold_value)
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

def threshold_image_standard(input_image,thresh):
     
    binary_image=input_image>thresh

    return binary_image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 
    
        
# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels

def coincidence_analysis_pixels(binary_image1,binary_image2):
    pixel_overlap_image=binary_image1&binary_image2         
    pixel_overlap_count=pixel_overlap_image.sum()
    pixel_fraction=pixel_overlap_image.sum()/binary_image1.sum()
    
    return pixel_overlap_image,pixel_overlap_count,pixel_fraction

# Look at coincidence in terms of features. Needs binary image input 

Output_all_cases = pd.DataFrame(columns=['Path','Number_of_events','Intensity_mean','Intensity_SD','Intensity_med',
                                       'Area_mean','Area_sd','Area_med','Length_mean','Length_sd','Length_med','Ratio_mean','Ratio_sd','Ratio_med'])



for path in pathList:
    print(path)
    path=path+"/"
    j=0
    for root, dirs, files in os.walk(path):
            for name in files:
                    if filename_contains in name:
                        if ".tif" in name:
                            resultsname = name
                            print(resultsname)
                            
                            
                            path=path
                            image_path=path+resultsname
                            
                            #  Make a new folder for images etc.
                            path_to_save=path+str(j)+"/"
                            
                          
                            
                            try: 
                                os.mkdir(path_to_save) 
                            except OSError as error: 
                                print(error) 
                            
                                
                            # Load the image: 
                            green=load_image(image_path)
                         
                            # z-project the image  
                        
                            green_flat=np.mean(green,axis=0)
                        
                            # Subtract the background
                        
                            green_filtered=subtract_bg(green_flat)
                        
                            # Save the images
                            imsr2 = Image.fromarray(green_flat)
                            imsr2.save(path_to_save+'Flat.tif')

                            imsr2 = Image.fromarray(green_filtered)
                            imsr2.save(path_to_save+'Filtered.tif')
                            
                            # Threshold the image:
                            green_threshold,green_binary=threshold_image_alt(green_filtered)
                            
                            # Save the binary image
                            im = Image.fromarray(green_binary)
                            im.save(path_to_save+'Binary.tif')
                            
                            # Perform analysis on these
                            green_number,green_labelled=label_image(green_binary)
                            print("%d feautres were detected in the green image."%green_number)
                            measurements=analyse_labelled_image(green_labelled,green_filtered)
                            
                            im = Image.fromarray(green_labelled)
                            im.save(path_to_save+'labelled.tif')
                            
                            labeltot=green_labelled.max()+1
                                
                            print('Total number of clusters in labelled image: %d'%labeltot)
                            
                            # Make and save histograms
                                    
                            areas=measurements['area']*((Pixel_size/1000)**2)
                            plt.hist(areas, bins = 20,range=[0,2], rwidth=0.9,color='#ff0000')
                            plt.xlabel('Area (\u03bcm$^2$)',size=20)
                            plt.ylabel('Number of Features',size=20)
                            plt.title('Cluster area',size=20)
                            plt.savefig(path_to_save+"Area.pdf")
                            plt.show()
                            
                            median_area=areas.median()
                            mean_area=areas.mean()
                            std_area=areas.std()
                            
                            
                            length=measurements['major_axis_length']*((Pixel_size))
                            plt.hist(length, bins = 20,range=[0,5000], rwidth=0.9,color='#ff0000')
                            plt.xlabel('Length (nm)',size=20)
                            plt.ylabel('Number of Features',size=20)
                            plt.title('Cluster lengths',size=20)
                            plt.savefig(path_to_save+"Lengths.pdf")
                            plt.show()
                        
                            median_length=length.median()
                            mean_length=length.mean()
                            std_length=length.std()
                            
                            ratio=measurements['minor_axis_length']/measurements['major_axis_length']
                            plt.hist(ratio, bins = 50,range=[0,1], rwidth=0.9,color='#ff0000')
                            plt.xlabel('Eccentricity',size=20)
                            plt.ylabel('Number of Features',size=20)
                            plt.title('Cluster Eccentricity',size=20)
                            plt.savefig(path_to_save+"Ecc.pdf")
                            plt.show()
                            
                            
                            intensities_all=measurements['max_intensity']
                            mean_intensity=intensities_all.mean()
                            std_intensity=intensities_all.std()
                            median_intensity=intensities_all.median()
                                                    
                            median_ratio=ratio.median()
                            mean_ratio=ratio.mean()
                            std_ratio=ratio.std()
                            
                            measurements['Eccentricity']=ratio
                           
                            
                            measurements.to_csv(path_to_save+ '/' + 'ThT_Metrics.csv', sep = '\t')
                            
                        
                            
                            Output_all_cases = Output_all_cases.append({'Path':path,'Number_of_events':labeltot,'Intensity_mean':mean_intensity,'Intensity_SD':std_intensity,'Intensity_med':median_intensity,
                                                                        'Area_mean':mean_area,'Area_sd':std_area,'Area_med':median_area,'Length_mean':mean_length,'Length_sd':std_length,'Length_med':median_length,
                                                                        'Ratio_mean':mean_ratio,'Ratio_sd':std_ratio,'Ratio_med':median_ratio},ignore_index=True)
                        
                        
                            Output_all_cases.to_csv(root_path + 'all_DL_metrics.csv', sep = '\t')
                        
                            j+=1

