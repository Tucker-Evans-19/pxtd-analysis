import numpy as np
import matplotlib.pyplot as plt
import h5py as hp
import sys
from scipy import ndimage
import scipy.linalg as lin
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from scipy.optimize import least_squares
import numpy.linalg as linalg
import os
import matplotlib.patches as pat
import pandas as pd
import scipy.special as sp
from scipy.linalg import convolution_matrix
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import plasma_analysis.spectraProp as sp
#import pxtd_analysis as pa
import scipy.optimize as so
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy.random as rd
import plasma_analysis.ballabio as ba


#------------------------------#
# Parameter definitions:
# -----------------------------#
#scintillator response
risetime = 20
falltime = 1000
inter_fid_time = 548.25

# window on PXTD lineout that we are considering
fidlow = 82
fidhigh = 92
fidcenter = 87

left_side = 100
right_side = 430

# defining the shape of the median filter
med_filt2 = 1
med_filt1 = 10

#-------------------------------------------------------------------
# basic function definitions
#-------------------------------------------------------------------
def timeAxis(x,a,b,c):
    # we need to have a fit to the timing fiducials.
    # according to Neel and Hong's work this can just be a quadratic
    return a*x**2 + b*x + c

def Gauss(x, A, B, x0):
	y = A*np.exp(-(x-x0)**2/(2*B**2))
	return y

def normal_gaussian(x, B, x0):
    return np.exp(-(x-x0)**2/(2*B**2))

def Two_Gauss(x, A1, B1, x1, A2, B2, x2):	
	y = A1*np.exp(-1*(x-x1)**2/(2*B1**2)) + A2*np.exp(-1*(x-x2)**2/(2*B2**2))
	return y

def skew_gaussian(x, A, B, x0, alpha):
    y = (1/np.sqrt(2*np.pi)) * Gauss(x, A, B, x0)*.5*(1 + sp.erf(alpha * (x-x0)/(sqrt(2)*2*B)))
    return y

def scint_IRF(t, t_rise, t_fall):
    irf_unnorm = (1-np.exp(-t/t_rise)) * np.exp(-t/t_fall)
    irf_norm = irf_unnorm/np.max(irf_unnorm)
    return irf_norm

def pxtd_IRF(t, t_rise, t_fall):
    return scint_IRF(t, t_rise, t_fall)


def pxtd_IRF_conv_mat(t, t_rise, t_fall, length):
    return convolution_matrix(pxtd_IRF(t, t_rise, t_fall), length, mode = 'same')

def convolved_signal(conv_mat, emission_history):
    return np.matmul(conv_mat, emission_history)

def roll_average(vector_input, half_window):
	vector_averaged = vector_input
	for i in range(len(vector_averaged)):
		num_elements = 0
		average_sum = 0
		for w in range(i - half_window, i+half_window):
			if w >= 0 and w<len(vector_averaged):
				average_sum += vector_averaged[w]
				num_elements += 1
		vector_averaged[i] = average_sum/num_elements
	return vector_averaged


#--------------------------------------------------
# MAIN FUNCTIONS FOR ANALYZING A RAW PXTD H5 FILE: 
#--------------------------------------------------


def show_pxtd_image(file):
    
    #breaking out file information:
    shot_num = file[-13:-7]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype = float)
    #image_clean = image_raw[0,:,:]*image_raw[1,:,:]**-1
    image_clean = image_raw[0,:,:]-image_raw[1,:,:]
    #image_filter = image_clean <= 6000
    #image_clean = image_filter*image_clean
    
    #rotating the image (needed for all PXTD images)
    
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
    #image_rot = image_rot[65:230, 40:460]
    filtered_image = ndimage.median_filter(image_rot, size = 3)
    plt.imshow(filtered_image, cmap = 'binary')
    plt.colorbar()
    plt.title(f'PXTD image, shot {shot_num}')
    
    image = filtered_image
    plt.grid()
    
    return image, shot_num

def show_ntd_image(file):
    
    #breaking out file information:
    shot_num = file[-16:-10]
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype = float)
    
    #subtracting off the BG
    image_clean = image_raw[0,:,:]-image_raw[1,:,:]
    #image_clean = ndimage.rotate(image_clean, -.0138*180/3.1415)
    filtered_image = ndimage.median_filter(image_clean, size = 3)
    plt.imshow(np.log(filtered_image), cmap = 'binary')
    plt.colorbar()
    plt.title(f'PXTD image, shot {shot_num}')
    plt.grid()
    
    image = filtered_image
    
    return image, shot_num


def pxtd_lineout(file, channel_center, channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):

    c1low = int(channel_center - np.ceil(channel_width/2))
    c1high = int(channel_center + np.ceil(channel_width/2))
    #---------------------------------------------------------------
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    if shot_num ==  -1:
        shot_num = file[-13:-7]

    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype=float)
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    image_clean = ndimage.rotate(image_clean, .0297*180/3.1415)

    if left_side == -1:
        left_side = 0
    if right_side == -1:
        right_side = image_clean.shape[1]
    

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_clean[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    #plt.figure()
    #plt.pcolor(c1_lineout_block)
    #plt.show()

    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    if bg_subtract == True:
        bg_filter = (c1_lineout<.04*np.max(c1_lineout))
        bg_average = np.sum(c1_lineout*bg_filter)/np.sum(bg_filter)
        c1_lineout -= bg_average

    if plotting == True:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(c1_lineout_block)
        ax[1].plot(c1_lineout)


    return c1_lineout


def ntd_lineout(file, channel_center, channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):

    c1low = int(channel_center - np.ceil(channel_width/2))
    c1high = int(channel_center + np.ceil(channel_width/2))
    #---------------------------------------------------------------
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    if shot_num ==  -1:
        shot_num = file[-13:-7]

    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype=float)
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]

    if left_side == -1:
        left_side = 0
    if right_side == -1:
        right_side = image_clean.shape[1]
    

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_clean[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    #plt.figure()
    #plt.pcolor(c1_lineout_block)
    #plt.show()

    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    if bg_subtract == True:
        bg_filter = (c1_lineout<.04*np.max(c1_lineout))
        bg_average = np.sum(c1_lineout*bg_filter)/np.sum(bg_filter)
        c1_lineout -= bg_average

    if plotting == True:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(c1_lineout_block)
        ax[1].plot(c1_lineout)


    return c1_lineout


def p510_lineouts(file, channel_centers = [29, 495, 77, 117, 161, 204, 245, 286, 327, 368, 412, 454, 496], channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(ntd_lineout(file, channel_center, channel_width = channel_width, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
    return lineouts


def pxtd_3ch_lineouts(file, channel_centers = [146, 177, 213], channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(pxtd_lineout(file, channel_center, channel_width = channel_width, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
    return lineouts

def pxtd_2ch_lineouts(file, channel_centers = [161,193], channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(pxtd_lineout(file, channel_center, channel_width = channel_width, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
    return lineouts

def ntd_4ch_lineouts(file, channel_centers = [316, 480, 615, 770], channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(ntd_lineout(file, channel_center, channel_width = 30, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
    return lineouts

def get_ntd_fid_timing(file, channel_width = 10, plotting = False, shot_num = -1, left_side = 0, right_side = 1024, left_padding = 0, center = 165):
    fid_lineout = ntd_lineout(file, center, channel_width = 30)

    if left_padding > 0:
        fid_lineout = np.concatenate((np.zeros(left_padding-1), fid_lineout))
    #fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (10, 10))
    fid_lineout = ndimage.median_filter(fid_lineout, size = (10))
    #fid_lineout = np.log(fid_lineout-np.min(fid_lineout) + 1)
    #fid_lineout = ndimage.median_filter(fid_lineout, size = (10))

    manual_timing = False
    if manual_timing:
        plt.figure()
        plt.plot(fid_lineout)
        plt.show()
        input_num_peaks = int(input('input the number of peaks: ') )
        print('click on the peaks of the fiducials')
        plt.figure()
        plt.plot(fid_lineout)
        clicks = plt.ginput(input_num_peaks)
        plt.close()
        

        plt.figure()
        peak_width = 90
        print(clicks)
        centers = []
        for click in clicks:
            peak_top = int(click[0] + peak_width/2)
            peak_bottom = int(click[0] - peak_width/2)
            print(peak_top)
            print(peak_bottom)
            plt.plot(np.arange(peak_bottom, peak_top), fid_lineout[peak_bottom:peak_top]/np.max(fid_lineout[peak_bottom:peak_top]), color = 'green')
            params, cov = so.curve_fit(normal_gaussian, np.arange(peak_bottom, peak_top), fid_lineout[peak_bottom:peak_top]/np.max(fid_lineout[peak_bottom:peak_top]), p0 = [40, click[0]])
            print(params[1])
            centers.append(params[1])
    else:
        centers, properties = ss.find_peaks(fid_lineout, height = np.max(fid_lineout)/2) 
    plt.vlines(centers, ymin = 0, ymax = 1)

    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, inter_fid_time*np.arange(len(centers)))
    a,b,c = t_params
    time_axis = (timeAxis(np.arange(len(fid_lineout)),a,b,c))
    time_axis -= min(time_axis)
    time_per_pixel = inter_fid_time/average_peak_diff
    
    return time_axis, centers, fid_lineout

def get_pxtd_fid_timing(file, channel_width = 30, plotting = False, shot_num = -1, left_side = -1, right_side = -1, left_padding = 0, center = fidcenter):
    fid_lineout = pxtd_lineout(file, center, channel_width = channel_width)

    if left_padding > 0:
        fid_lineout = np.concatenate((np.zeros(left_padding-1), fid_lineout))
    #fid_lineout_block = ndimage.median_filter(image_rot[fidlow:fidhigh, left_side:right_side], size = (10, 10))
    fid_lineout = ndimage.median_filter(fid_lineout, size = (10))
    #fid_lineout = np.log(fid_lineout-np.min(fid_lineout) + 1)
    #fid_lineout = ndimage.median_filter(fid_lineout, size = (10))
    plt.figure()
    plt.plot(fid_lineout)
    plt.show()
    input_num_peaks = int(input('input the number of peaks: ') )
    print('click on the peaks of the fiducials')
    plt.figure()
    plt.plot(fid_lineout)
    clicks = plt.ginput(input_num_peaks)
    plt.close()
    

    plt.figure()
    peak_width = 20
    print(clicks)
    centers = []
    for click in clicks:
        peak_top = int(click[0] + peak_width/2)
        peak_bottom = int(click[0] - peak_width/2)
        print(peak_top)
        print(peak_bottom)
        plt.plot(np.arange(peak_bottom, peak_top), fid_lineout[peak_bottom:peak_top]/np.max(fid_lineout[peak_bottom:peak_top]), color = 'green')
        params, cov = so.curve_fit(normal_gaussian, np.arange(peak_bottom, peak_top), fid_lineout[peak_bottom:peak_top]/np.max(fid_lineout[peak_bottom:peak_top]), p0 = [40, click[0]])
        print(params[1])
        centers.append(params[1])
    plt.vlines(centers, ymin = 0, ymax = 1)

    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)

    t_params, t_cov = curve_fit(timeAxis, centers, inter_fid_time*np.arange(len(centers)))
    a,b,c = t_params
    time_axis = (timeAxis(np.arange(len(fid_lineout)),a,b,c))
    time_axis -= min(time_axis)
    time_per_pixel = inter_fid_time/average_peak_diff
    
    return time_axis, centers, fid_lineout

def deconvolve_TOF(t, lineout, reaction, tion, distance, plotting = False):
    # this assumes that you have already deconvolved the IRF and are providing a lineout that should only correspond to TOF differences.
    # creating the energy spread matrix 
    filter_mat = (lineout >= np.max(lineout)*.01)
    e_spread_mat = energy_spread_matrix(t, reaction, tion, distance)
    def objective(emission_fit):
        return (np.matmul(e_spread_mat, emission_fit) - lineout)* filter_mat
    result = so.least_squares(objective, np.zeros(shape = lineout.shape), bounds = (0, np.inf), method = 'dogbox', max_nfev = 200000)
    emission_fit = result.x
    signal_fit = np.matmul(e_spread_mat, emission_fit)
    residual = lineout - signal_fit
    return t, emission_fit, signal_fit, residual

def deconvolve_IRF(t, lineout, plotting = False, prepend = True, num_bins = 500, filterLeft = -1, filterRight = -1):
    # interpolating data to an even grid for the fitting
    if prepend == True:
        time_axis = np.linspace(np.min(t)-1400, np.max(t), num_bins)
        time_axis_out = time_axis
    else:
        time_axis = t
        time_axis_out = time_axis
        
    
    data_interp = np.interp(time_axis, t, lineout, left=0) #interpolating data to intermediate grid
    selection_filter = np.zeros(time_axis.shape)
    if (filterLeft == -1) or (filterRight == -1): 
        plt.figure()
        plt.plot(data_interp)
        print('click to the left and then to the right of the region to be analyzed.')
        left_click, right_click = plt.ginput(2)
        plt.close()
        # generating a filter based on the selection of left and right clicks above
        filterLeft = int(left_click[0])
        filterRight = int(right_click[0])
    print(f'selection filter shape : {selection_filter.shape}')
    print(f'filterRight: {filterRight}')
    print(f'filterLeft: {filterLeft}')

    selection_filter[filterLeft:filterRight] = np.ones(filterRight - filterLeft)

    A = pxtd_conv_matrix(time_axis)
    
    #defining the objective function to be minimized in fitting the emission history. 
    def objective(x):
        return ((np.matmul((A), x) - data_interp + data_interp[0]))*selection_filter
    result = so.least_squares(objective, np.abs(np.gradient(data_interp)), method = 'dogbox', max_nfev = 40000, bounds = (0, np.inf),xtol = 5e-11)
    emission_fit = result.x
    signal_fit = np.matmul(A, emission_fit)
    residual = data_interp - signal_fit
    return time_axis_out, emission_fit, signal_fit, residual


#=================================================================================================
# The functions below are for the creation of synthetic spectra and for their propagation in
# time to recreate observed spectra at a detector.
'''
# UNITS:
 * energy, MeV
 * velocity, cm/s
 * temperature, keV

'''
#=================================================================================================
# DEFINING SOME IMPORTANT CONSTANTS:

rm_proton = 938.272 # MeV
rm_neutron  = 939.565 # MeV
c = .02998 # cm/ps

# from Hong's thesis
#rise_time = .089
#fall_time = 1.4

# from discussion with Neel
rise_time = 20
fall_time = 900

# 
rise_time = 20
fall_time = 1000

#---------------------------------------------------------------------------------------------
def plot_spectrum(bin_centers, counts, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(bin_center, counts)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Counts')


def synth_spec_gauss(particle_type, temperature, num_particles=5000, birth_time = 0):
    #ballabio calculation of the mean energy and standard deviation
    mean_energy, sigma_energy =  ba.ballabio_mean_std(particle_type, temperature)

    # creating synthetic population
    pop = rd.normal(mean_energy, sigma_energy, num_particles)


    return pop, mean_energy, sigma_energy

def get_pop_velocities(particle_type, pop_energies):
    
    # getting the velocity distribution:
    if particle_type == 'DTn':
        mass = rm_neutron
    elif particle_type == 'DDn':
        mass = rm_neutron
    elif particle_type == 'D3Hep':
        mass = rm_proton
    elif particle_type == 'xray':
        velocities = np.ones(pop_energies.shape)*c
        return velocities
    else:
        particle_type = 'None'
        mass = 0 
        print('invalid particle type')

    ER = pop_energies/mass
    #print(ER)
    beta2 = 1 - (ER + 1)**-2
    #print(beta2)
    beta = beta2**.5
    #print(beta)
    velocities = beta*c

    return velocities


def time_to_dist(dist, velocities):
    times = dist * velocities**-1
    return times
        
def pop_histogram(pop, ax = None):
    num_numparticles = pop.size
    
    if ax ==None:
        fig, ax = plt.subplots()
    energy_bin_edges = np.linspace(0, 20, min([num_particles/2, 200]))
    counts, energy_bins = ax.hist(pop, energy_bin_edges)
    ax.set_xlim([10, 20])
    return counts, energy_bins

def time_trace_at_dist(dist, particle_type, population_energies, birth_time, time_bins):
    velocities  = get_pop_velocities(particle_type, population_energies)
    times = time_to_dist(dist, velocities) + birth_time
    counts_per_time, time_bins = np.histogram(times, bins = time_bins)
    return time_bins, counts_per_time

def energy_spread_matrix(time_axis, particle_type, temperature, distance):
    # for a given type of particle at a given temperature, return a matrix A such that Ax is
    # the fluence(time) through a detector placed at a distance d from the source 
    
    # instantiating an empty version of our matrix
    num_steps = time_axis.size
    energy_spread_matrix = np.zeros((num_steps, num_steps))
    time_list = list(time_axis)
    time_step = time_axis[2] - time_axis[1]
    time_list.append(time_axis[-1] + time_step) 
    time_bin_edges = np.asarray(time_list)

    # allowing for energies 3 sigma to either side of the mean_energy
    #energy_axis = np.linspace(mean_energy - 4*sigma_energy, mean_energy + 3*sigma_energy, num_energy_bins)

    # calculating number of particles to have the energy specified (normalized spectrum)
    #num_per_energy = np.exp(-(energy_axis - mean_energy)**2/(2*sigma_energy**2))
    #num_per_energy = num_per_energy*np.sum(num_per_energy)**-1
    if particle_type != 'xray':
        population_energies, mean_energy, sigma_energy = synth_spec_gauss(particle_type, temperature)
    else:
        population_energies = np.ones(5000)
    time_bin_edges, time_trace = time_trace_at_dist(distance, particle_type, population_energies, 0, time_bin_edges)
    time_trace = time_trace/np.sum(time_trace)
    for i in range(num_steps):
        for j in range(num_steps):
            if j>=i:
                energy_spread_matrix[i,j] = time_trace[j-i]
            else:
                energy_spread_matrix[i,j] = 0

    return np.transpose(energy_spread_matrix)
    
def pxtd_conv_matrix(bin_centers, plotting = False):
    # creates a matrix G such that Gy is the recorded history of the PXTD detector given a fluence at time t of y
    num_counts = bin_centers.size
    G = np.zeros(shape = (num_counts, num_counts))
    for i in range(num_counts):
        for j in range(num_counts):
            if j>=i:
                #G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*np.exp(-(bin_centers[j] - bin_centers[i])/fall_time)
                G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*(np.exp(-(bin_centers[j] - bin_centers[i])/900) + .07*np.exp(-(bin_centers[j] - bin_centers[i])/12000))
                #G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*np.exp(-(bin_centers[j] - bin_centers[i])/fall_time)+(1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*np.exp(-(bin_centers[j] - bin_centers[i]-rise_time)/(fall_time*1.6))
            else:
                G[i,j] = 0
    G=G*np.sum(G[0, :])**-1
    if plotting ==True:
        plt.figure()
        plt.imshow(G)
    
    return np.transpose(G) 

def pxtd_IRF(time_axis):
    IRF = (1-np.exp(-(time_axis)/rise_time))*(np.exp(-(time_axis)/900) + .07*np.exp(-(time_axis)/12000))
    IRF /= np.sum(np.sqrt(IRF**2))
    return IRF
'''
def pxtd_conv_IRF(bin_centers, plotting = False):
    # creates a matrix G such that Gy is the recorded history of the PXTD detector given a fluence at time t of y
    num_counts = bin_centers.size
    IRF = np.zeros(shape = (num_counts,))
    for j in range(num_counts-1):
        IRF[j] = (1-np.exp(-(bin_centers[j+1] - bin_centers[0]-20)/rise_time))*np.exp(-(bin_centers[j+1]-bin_centers[0]-20)/fall_time)
    IRF[-1] = IRF[-2]
    IRF=10000*IRF*np.sum(IRF)**-1
    for j in range(num_counts):
        if IRF[j] <= .02:
            IRF[j] = .02

    if plotting ==True:
        plt.figure()
        plt.plot(bin_centers, IRF)
    
    return IRF 

def pxtd_lineout(file, channel_center, channel_width = 10, plotting = False, shot_num = -1, left_side = 0, right_side = -1, bg_subtract = True):

    c1low = int(channel_center - np.ceil(channel_width/2))
    c1high = int(channel_center + np.ceil(channel_width/2))
    #---------------------------------------------------------------
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    if shot_num ==  -1:
        shot_num = file[-13:-7]

    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype=float)
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    
    #rotating the image (needed for all PXTD images)
    image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
    image_rot = ndimage.median_filter(image_rot, size = 3)
    
    if right_side == -1:
        right_side = image_rot.shape[1]
    
    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = ndimage.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    
    #averaging to find lineout of each channel
    c1_lineout = np.average(c1_lineout_block, 0)
    if bg_subtract == True:
        bg_filter = (c1_lineout<.04*np.max(c1_lineout))
        bg_average = np.sum(c1_lineout*bg_filter)/np.sum(bg_filter)
        c1_lineout -= bg_average

    if plotting == True:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(c1_lineout_block)
        ax[1].plot(c1_lineout)


    return c1_lineout
'''
