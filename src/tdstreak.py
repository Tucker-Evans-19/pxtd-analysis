# plotting stuff
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# scipy stuff
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from scipy.optimize import least_squares
import scipy.optimize as so
import scipy.ndimage as nd
import scipy.signal as ss
import scipy.special as sp
import scipy.linalg as lin
from scipy.linalg import convolution_matrix
from scipy.signal import welch

# numpy stuff
import numpy.random as rand
import numpy as np
import numpy.random as rd
from numpy.fft import fft, ifft, fftfreq
import numpy.linalg as linalg

# system stuff
import os
import sys

#my packages
import plasma_analysis.ballabio as ba

# data handling
import pandas as pd
import h5py as hp
import re

#------------------------------#
# PXTD and Scintillator Parameter definitions:
# -----------------------------#
#scintillator response
risetime = 20
falltime = 1000
inter_fid_time = 548.25

#fiducial window definitions for ntd:
fidlow = 82
fidhigh = 92
fidcenter = 87

left_side = 100
right_side = 430
#final_lineout_index = 1300

# defining the shape of the median filter
med_filt2 = 1
med_filt1 = 10

#=================================================================================================
# DEFINING IMPORTANT CONSTANTS FOR SPECTRAL PROPAGATION:
#=================================================================================================

rm_proton = 938.272 # MeV
rm_neutron  = 939.565 # MeV
c = .02998 # cm/ps

# from Hong's thesis
#rise_time = .089
#fall_time = 1.4

# from discussion with Neel
#rise_time = 20
#fall_time = 900

# 
#rise_time = 20
#fall_time = 1000

rise_time = 20
#nominal values:
fall1 = 900
fall2 = 12000
fall2_weight = .07
'''
fall1 = 850
fall2 = 14100
fall2_weight = .07
'''
W_rise_time = 10
W_fall1 = 100

#=================================================================================================
# FUNCTION DEFINITIONS
#=================================================================================================

## TIMING AND BASIC SHAPE FUNCTIONS 

# quadratic function that can be used as a time axis for streak camera data.
def timeAxis(x,a,b,c):
    # we need to have a fit to the timing fiducials.
    # according to Neel and Hong's work this can just be a quadratic
    return a*x**2 + b*x + c

# single gaussian function
def Gauss(x, A, B, x0):
	y = A*np.exp(-(x-x0)**2/(2*B**2))
	return y

# gaussian with A=1, useful for normalized data
def normal_gaussian(x, B, x0):
    return np.exp(-(x-x0)**2/(2*B**2))

# two gaussians added together
def Two_Gauss(x, A1, B1, x1, A2, B2, x2):	
	y = A1*np.exp(-1*(x-x1)**2/(2*B1**2)) + A2*np.exp(-1*(x-x2)**2/(2*B2**2))
	return y

# skew gaussian, more appropriate for time-dep output ICF implosion (shock component or compression component)
def skew_gaussian(x, A, B, x0, alpha):
    y = (1/np.sqrt(2*np.pi)) * Gauss(x, A, B, x0)*.5*(1 + sp.erf(alpha * (x-x0)/(np.sqrt(2)*2*B)))
    return y

## IMPULSE RESPONSE FUNCTIONS 

# scintillator impulse response function for either the NTD or PTD diagnostics
def scint_IRF(t, t_rise, t_fall):
    irf_unnorm = (1-np.exp(-t/t_rise)) * np.exp(-t/t_fall)
    irf_norm = irf_unnorm/np.max(irf_unnorm)
    return irf_norm

def gaussian_spread(t, width, t0 = 0): 
    G = np.exp(-(t-t0)**2/(2*width**2)) # producing a spread function (gaussian)
    G /= G.sum() # normalizing
    return G
    
# specifically the impulse response function of PXTD diagnostic
#def pxtd_IRF(t):
#    return scint_IRF(t, pxtd_rise, pxtd_fall)

# returns the convolution matrix for the PXTD diagnostic
# for a given output vs time, the mult of this matrix with your time signal will give you the detectors response vs time
def pxtd_IRF_conv_mat(t, t_rise, t_fall, length):
    return convolution_matrix(pxtd_IRF(t, t_rise, t_fall), length, mode = 'same')

def pxtd_conv_matrix(bin_centers, plotting = False):
    # creates a matrix G such that Gy is the recorded history of the PXTD detector given a fluence at time t of y
    num_counts = bin_centers.size
    G = np.zeros(shape = (num_counts, num_counts))
    for i in range(num_counts):
        for j in range(num_counts):
            if j>=i:
                G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/rise_time))*(np.exp(-(bin_centers[j] - bin_centers[i])/fall1) + fall2_weight*np.exp(-(bin_centers[j] - bin_centers[i])/fall2))
            else:
                G[i,j] = 0
    G=G*np.sum(G[0, :])**-1
    if plotting ==True:
        plt.figure()
        plt.imshow(G)
    
    return np.transpose(G) 

def pxtd_IRF(time_axis):
    IRF = (1-np.exp(-(time_axis)/rise_time))*(np.exp(-(time_axis)/fall1) + fall2_weight*np.exp(-(time_axis)/fall2))
    IRF /= np.sum(np.sqrt(IRF**2))
    return IRF

def W_IRF(time_axis):
    IRF = (1-np.exp(-(time_axis)/W_rise_time))*(np.exp(-(time_axis)/W_fall1))
    IRF /= np.sum(np.sqrt(IRF**2))
    return IRF

def W_conv_matrix(bin_centers, plotting = False):
    
    num_counts = bin_centers.size
    G = np.zeros(shape = (num_counts, num_counts))
    for i in range(num_counts):
        for j in range(num_counts):
            if j>=i:
                G[i,j] = (1-np.exp(-(bin_centers[j] - bin_centers[i])/W_rise_time))*(np.exp(-(bin_centers[j] - bin_centers[i])/W_fall1))
            else:
                G[i,j] = 0
    G=G*np.sum(G[0, :])**-1
    if plotting ==True:
        plt.figure()
        plt.imshow(G)
    
    return np.transpose(G) 

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
'''

def convolved_signal(conv_mat, emission_history):
    return np.matmul(conv_mat, emission_history)

# return a rolling average for a given signal and window size. window is centered
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

#=================================================================================================
# PLOTTING STREAK CAMERA IMAGES#
#=================================================================================================

# for an inputted h5 file this will plot the corresponding streak detector data
def show_image(file, rotation_angle = 0, colormap = 'binary', num_edge_l=-13, num_edge_r=-7, scaling = 'linear', grid = True):
    
    
    #breaking out file information:
    #shot_num = file[num_edge_l:num_edge_r]
    shot_nums = (re.findall(r'\d+', file))
    print(shot_nums)
    for element in shot_nums:
        if float(element) > 10000:
            shot_num = element
            
        
    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype = float)
    if scaling == 'log':
        image_raw = np.log(image_raw)
    
    # subtracting the background of the image
    image_clean = image_raw[0,:,:]-image_raw[1,:,:]
    
    #rotating the image (needed for all PXTD images)
    #image_rot = ndimage.rotate(image_clean, .0297*180/3.1415)
    image_rot = nd.rotate(image_clean, float(rotation_angle))
    
    # median filtering the image to get rid of any spikes from direct neutron noise
    filtered_image = nd.median_filter(image_rot, size = 3)

    #plotting the data
    plt.imshow(filtered_image, cmap = colormap)
    plt.colorbar()
    plt.title(f'PXTD image, shot {shot_num}')
    plt.xlabel('Time ----> ') 
    if grid:
        plt.grid()
    
    return filtered_image, shot_num

# plotting an ntd image specifically (no rotation angle required)
def show_ntd_image(file, colormap = 'binary', scaling = 'linear', grid = True):
    filtered_image, shot_num = show_image(file, colormap = colormap, num_edge_l=-16, num_edge_r = -10, scaling = scaling, grid = grid)
    return filtered_image, shot_num

# plotting a pxtd image (rotation required to correct image)
def show_pxtd_image(file,rotation_angle = float(.0297*180/3.1415), colormap = 'binary', scaling = 'linear'):
    filtered_image, shot_num = show_image(file, rotation_angle = rotation_angle, colormap = colormap, scaling = scaling)
    return filtered_image, shot_num

# base function for plotting lineouts of an image. defaults to 10 pixel width
def image_lineout(file, channel_center, channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True, rotation_angle = 0):

    clow = int(channel_center - np.ceil(channel_width/2))
    chigh = int(channel_center + np.ceil(channel_width/2))
    #---------------------------------------------------------------
    # FILE INTERPRETATION
    #---------------------------------------------------------------

    #breaking out file information:
    if shot_num ==  -1:
        shot_num = file[-13:-7]

    data = hp.File(file, 'r')
    image_raw = np.array(data.get('Streak_array'), dtype=float)
    image_clean = image_raw[0,:,:] - image_raw[1,:,:]
    image_clean = nd.rotate(image_clean, rotation_angle)
    #image_clean = ndimage.rotate(image_clean, .0297*180/3.1415)

    if left_side == -1:
        left_side = 0
    if right_side == -1:
        right_side = image_clean.shape[1]
    

    #taking blocks of full image that will be the basis of the lineouts once averaged
    c_lineout_block = nd.median_filter(image_clean[clow:chigh,left_side:right_side], size = (med_filt1, med_filt2))

    #averaging in the vertical (non-time axis) to find lineout of each channel
    c_lineout = np.average(c_lineout_block, 0)
    if bg_subtract == True:
        bg_filter = (c_lineout<.04*np.max(c_lineout))
        bg_average = np.sum(c_lineout*bg_filter)/np.sum(bg_filter)
        c_lineout -= bg_average

    if plotting == True:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(c_lineout_block)
        ax[1].plot(c_lineout)


    return c_lineout

#=================================================================================================
# PLOTTING IMAGE LINEOUTS #
#=================================================================================================

def pxtd_lineout(file, channel_center, channel_width = 30, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):

    lineout = image_lineout(file, channel_center, channel_width = channel_width, plotting = plotting, shot_num = shot_num, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract, rotation_angle = .0297*180/3.14159)

    return lineout

def ntd_lineout(file, channel_center, channel_width = 40, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):

    lineout = image_lineout(file, channel_center, channel_width = channel_width, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True, rotation_angle = 0)

    return lineout

def p510_lineouts(file, channel_centers = [29, 495, 77, 117, 161, 204, 245, 286, 327, 368, 412, 454, 496], channel_width = 10, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(image_lineout(file, channel_center, channel_width = channel_width, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
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

def ntd_4ch_lineouts(file, channel_centers = [316, 480, 615, 770], channel_width = 80, plotting = False, shot_num = -1, left_side = -1, right_side = -1, bg_subtract = True):
    
    lineouts = []
    for channel_center in channel_centers:
        lineouts.append(ntd_lineout(file, channel_center, channel_width = channel_width, left_side = left_side, right_side = right_side, bg_subtract = bg_subtract))
    return lineouts


#=================================================================================================
# PULLING TIMING FROM FIDUCIALS
#=================================================================================================

def get_ntd_fid_timing(file, channel_width = 30, center = 155):
    fid_lineout = nd.gaussian_filter(ntd_lineout(file, center, channel_width = channel_width), 10) 
    centers, properties = ss.find_peaks(fid_lineout, height = np.max(fid_lineout)*.2,distance = 50) 

    #finding the average distance between the timing fiducial peaks: 
    peak_diffs = []
    for element in range(1,len(centers)):
        peak_diffs.append(centers[element] - centers[element-1])

    average_peak_diff = np.average(peak_diffs)
    print(centers) 
    t_params, t_cov = curve_fit(timeAxis, centers, inter_fid_time*np.arange(len(centers)))

    #TODO: CHange this to take into account the actual fiducial times rather than average etalon
    a,b,c = t_params
    time_axis = (timeAxis(np.arange(len(fid_lineout)),a,b,c))
    time_axis -= min(time_axis)
    time_per_pixel = inter_fid_time/average_peak_diff
    
    return time_axis, centers, fid_lineout

def get_pxtd_fid_timing(file, channel_width = 30, plotting = False, shot_num = -1, left_side = 0, right_side = -1, left_padding = 0, center = fidcenter):

    fid_lineout = pxtd_lineout(file, center, channel_width = channel_width)
    if left_padding > 0:
        fid_lineout = np.concatenate((np.zeros(left_padding-1), fid_lineout))
    fid_lineout = nd.median_filter(fid_lineout, size = (10))

    manual_timing = False
    if manual_timing: # deprecated capability for cases where the signals are not consistent enough to be analyzed automatically
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
    #plt.vlines(centers, ymin = 0, ymax = 1)

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
# SYNTHETIC SPECTRA AND SPECTRA DECONVOLUTION
#=================================================================================================
# The functions below are for the creation of synthetic spectra and for their propagation in
# time to recreate observed spectra at a detector.
'''
# UNITS:
 * energy, MeV
 * velocity, cm/s
 * temperature, keV

'''
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
    pop = rd.normal(mean_energy, sigma_energy, int(num_particles))


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
        velocities = np.ones_like(pop_energies)*c
        return velocities
    else:
        particle_type = 'None'
        mass = 0 
        print('invalid particle type')
    '''
    ER = np.array(pop_energies)/mass
    #print(ER)
    beta2 = 1 - (ER + 1)**-2
    #print(beta2)
    beta = beta2**.5
    #print(beta)
    velocities = beta*c
    '''
    #velocities = c*(((pop_energies+mass)**2 - mass**2)/(pop_energies + mass)**2)**.5
    velocities = c*np.sqrt(1-(1/(pop_energies/mass + 1)**2)) 
    return velocities


def time_to_dist(dist, velocities):
    times = dist * velocities**-1
    return times
        
def pop_histogram(pop, ax = None):
    num_particles = pop.size
    
    if ax ==None:
        fig, ax = plt.subplots()
    energy_bin_edges = np.linspace(0, 20, min([num_particles/2, 500]))
    counts, energy_bins, _ = ax.hist(pop, energy_bin_edges)
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


def wiener_decon(signal, kernel, isnr):
    #signal_freqs, signal_power = ss.periodogram(signal)
    
    H = fft(kernel) 
    Y = fft(signal)

    G = Y*np.conj(H)/(H*np.conj(H) + isnr**2)
    deconvolved = np.real(ifft(G))
    return deconvolved 

def wiener_decon_freqdep(signal, kernel, SNR_freqs, iSNR):
    iSNR_reflected = np.hstack([np.flip(iSNR), iSNR]) 
    SNR_freqs_reflected = np.hstack([np.flip(-SNR_freqs), SNR_freqs])

    H = fft(kernel)
    H_freqs = fftfreq(kernel.size)
    S = fft(signal)
    S_freqs = fftfreq(signal.size)

    #fig, ax = plt.subplots(1,3)

    #ax[0].plot(H_freqs, H)
    #ax[1].plot(S_freqs, S)
    #ax[2].plot(SNR_freqs_reflected, iSNR_reflected) 

    iSNR_interp = np.interp(S_freqs, SNR_freqs_reflected, iSNR_reflected)
    #ax[2].plot(S_freqs, iSNR_interp)
    

    X = S*np.conj(H)/(H*np.conj(H)+ iSNR_interp)
    
    


    
    deconvolved = (ifft(X))
    return deconvolved 

def exponential(x,A,g):
    return A*np.exp(-g*x) 

def gaussian(x, A,x0,B):
    return A*np.exp(-(x-x0)**2/(2*B**2))

#def skew_gaussian(x, x0, s, a):
#    return (1/np.sqrt(2*np.pi))*.5*np.exp(-(x-x0)**2/(2*s**2))*(1+np.erf(a*(x-x0)/(2*s*np.sqrt(2))))

def extrapolate_back(input_lineout):

    plt.figure()


    #plotting the full signal so that the user can identify the leading edge.
    
    indices = range(input_lineout.size)
    plt.plot(indices, input_lineout)
    left_click, right_click = plt.ginput(2)
    print(left_click[0])
    print(right_click[0])
    left = int(left_click[0])
    right = int(right_click[0])
    plt.close()

    #using the points that we have chosen to identify the 'rising' edge signal and isolate it from the full signal

    rise_fit_data = input_lineout[left:right]
    rise_fit_indices = indices[left:right]
    plt.plot(rise_fit_indices, rise_fit_data)
    plt.close()

    #defining the fitting function for the beginning of the signal and fitting our region of interest

    def rise_func(x, A, x0, g):
        try:
            return A*np.exp(g*(x-x0))
        except(RuntimeError):
            return 0
    
    try:
        popt, pcov = so.curve_fit(rise_func, rise_fit_indices, rise_fit_data, bounds = (0, np.inf), maxfev = 4000)
    except(RuntimeError):
        popt = [0,0,0]
        pcov = [0,0,0]
    print(popt)
    
    #creating the new lineouts extrapolating back from the beginning of the signal

    new_indices = np.arange(-left_side_pad, np.max(indices))
    new_l1 = []
    for index in new_indices:
        if index<=left:
            new_l1.append(rise_func(index, popt[0], popt[1], popt[2]))
        else:
            new_l1.append(input_lineout[index])
        if np.isnan(new_l1[-1]):
            new_l1[-1] = 0
    return new_l1


# TODO: GET THROUGH THIS MAIN SECTION FOR ANALYZING A FILE: 

def extrapolate_back_lineouts(lineouts):
    print('FITTING LEADING EDGE........')
    new_lineouts = []
    
    for l_num in range(len(lineouts)):
        lineout = lineouts[l_num]
        new_lineout = extrapolate_back(lineout)
        new_lineouts.append(new_lineout)

    return lineouts

def wiener_decon_lineout_avnoise(time_axis, lineout, shotnum = 0, noise_left = 0, noise_right = 0, signal_left = 0, signal_right = 0, padding_length = 4000, manual = True):
    
    # CHOOSING DATA AND NOISE REGIONS 
    #------------------------------------------------------------------------------------------------------------------
    #Plot the lineout and ask the user to pick the relevant regions to do the fitting and evaluate the noise
    print(manual)
    if manual: 
        print('Working in manual mode to choose regions of interest...')
        plt.figure()
        plt.plot(lineout) #plotting without axis to make it easier to get the index
        plt.xlabel('time (pixels)')
        plt.ylabel('channel/filtering position (pixels)') 
        plt.title('Pick falling region for noise spectrum (left and right)') # this should be a region that is exponential
        
        clicks = plt.ginput(2)
        noise_left = int(clicks[0][0])
        noise_right = int(clicks[1][0])

        plt.title('Pick signal region (left and right)') # this should be the whole area that has signal
        clicks = plt.ginput(2)
        signal_left = int(clicks[0][0])
        signal_right = int(clicks[1][0])

    # SIGNAL DATA: 
    #------------------------------------------------------------------------------------------------------------------
    #pulling in the data only between the clicks
    signal_data = []
    signal_time = []
    for element in range(signal_left, signal_right):
        signal_data.append(lineout[element])
        signal_time.append(time_axis[element]) 

    signal_data2 = np.array(signal_data)**2
    # the average of the whole signal is taken as a baseline
    MSS = np.average(signal_data2)

    # NOISE DATA: 
    #------------------------------------------------------------------------------------------------------------------
    # pulling the noise data in the selected region and fitting an exponential
    # the difference from the curve is then binned to understand the power spectral density
    noise_data = []
    for element in range(noise_left, noise_right):
        noise_data.append(lineout[element])
    noise_axis = np.arange(0, len(noise_data))
    noise_popt, noise_pcov = so.curve_fit(exponential, noise_axis, noise_data)
    
    #fitting an exponential to the decay region to subtract away and get noise
    noise_fit = exponential(noise_axis, noise_popt[0], noise_popt[1])
    #plt.figure()
    #plt.plot(noise_axis, noise_data)
    #plt.plot(noise_axis, noise_fit)
    noise_diff = noise_data - noise_fit
    #plt.plot(noise_diff) 
    if shotnum != 0:
        plt.savefig(f'noiseFit_{shotnum}_{channel_num}.png')
        plt.close()

    # finding the mean squared noise value
    noise2 = noise_diff**2
    MSE = np.average(noise2) 
    SNR = (MSS/MSE)**.5
    SNR_inv = (MSE/MSS)**.5

    nps = 1000
    smoothed_signal = nd.median_filter(signal_data, 6)
    #signal_psd = welch(smoothed_signal, nperseg = nps)
    signal_psd = welch(smoothed_signal, nperseg =nps)
    noise_psd = welch(signal_data - smoothed_signal, nperseg = nps)
    
    SNR_freq = signal_psd[0]
    iSNR = np.sqrt(noise_psd[1]/signal_psd[1])

    print(f'Mean squared noise value: {MSE:.2f}') 
    print(f'Estimated signal to noise ratio = {SNR:.2f}')
    print(f'Estimaged inverse SNR = {SNR_inv:.2f}')

    # building a periodogram for the noise data:
    noise_freqs, noise_power = ss.periodogram(noise_diff - np.average(noise_diff)) 
    # building a periodogram for the signal:
    signal_freqs, signal_power = ss.periodogram(signal_data)
    
    # WIENER DECONVOLUTION: 
    #------------------------------------------------------------------------------------------------------------------
    # this uses the spectral information from above to perform the wiener deconvolution
    
    # the data has to be padded to the right since there is an artificial negative signal imposed otherwise
    average_time_step = np.average(np.diff(time_axis)) 
    
    padding_axis_right = np.arange(1,padding_length)
    final_value = lineout[signal_right]
    
    padding_axis_left = np.arange(1,padding_length)
    initial_value = lineout[signal_right]
    # creating the time axis and padding data to the right side of the lineout
    signal_padding_right= final_value*(np.exp(-average_time_step*padding_axis_right/fall1) + fall2_weight*np.exp(-average_time_step*padding_axis_right/fall2))
    signal_padding_left= np.zeros_like(padding_axis_left)

    time_padding_right = padding_axis_right*average_time_step
    time_padding_left = -np.flip(padding_axis_left*average_time_step)

    # appending the padding data to the right 
    signal_padded = np.hstack((signal_padding_left, lineout[:signal_right], signal_padding_right))
    signal_time_padded = np.hstack((time_padding_left + time_axis[0], time_axis[:signal_right], time_padding_right+time_axis[signal_right]))
    shift = signal_time_padded.min()
    signal_time_padded -= signal_time_padded.min()
    # creating the IRF for the PXTD detector
    IRF = pxtd_IRF(np.array(signal_time_padded))
    #IRF = nd.gaussian_filter(IRF, 50)
    decon = wiener_decon(signal_padded, IRF, SNR_inv)
    #decon = wiener_decon_freqdep(signal_padded, IRF, SNR_freq, iSNR)
    #decon = wiener_decon_freqdep(signal_data, IRF, welch(noise_data, nperseg = 40)[1])

    return signal_time_padded[padding_length:-padding_length]+shift, decon[padding_length:-padding_length]

def wiener_decon_lineout(time_axis, lineout, shotnum = 0, noise_left = 0, noise_right = 0, signal_left = 0, signal_right = 0, padding_length = 4000, manual = True):
    
    # CHOOSING DATA AND NOISE REGIONS 
    #------------------------------------------------------------------------------------------------------------------
    #Plot the lineout and ask the user to pick the relevant regions to do the fitting and evaluate the noise
    print(manual)
    if manual: 
        print('Working in manual mode to choose regions of interest...')
        plt.figure()
        plt.plot(lineout) #plotting without axis to make it easier to get the index
        plt.xlabel('time (pixels)')
        plt.ylabel('channel/filtering position (pixels)') 
        plt.title('Pick falling region for noise spectrum (left and right)') # this should be a region that is exponential
        
        clicks = plt.ginput(2)
        noise_left = int(clicks[0][0])
        noise_right = int(clicks[1][0])

        plt.title('Pick signal region (left and right)') # this should be the whole area that has signal
        clicks = plt.ginput(2)
        signal_left = int(clicks[0][0])
        signal_right = int(clicks[1][0])

    # SIGNAL DATA: 
    #------------------------------------------------------------------------------------------------------------------
    #pulling in the data only between the clicks
    signal_data = []
    signal_time = []
    for element in range(signal_left, signal_right):
        signal_data.append(lineout[element])
        signal_time.append(time_axis[element]) 

    signal_data2 = np.array(signal_data)**2
    '''
    signal_data = lineout
    signal_data2 = np.array(signal_data)**2
    signal_time = time_axis
    '''
    # the average of the whole signal is taken as a baseline
    MSS = np.average(signal_data2)

    # NOISE DATA: 
    #------------------------------------------------------------------------------------------------------------------
    # pulling the noise data in the selected region and fitting an exponential
    # the difference from the curve is then binned to understand the power spectral density
    noise_data = []
    for element in range(noise_left, noise_right):
        noise_data.append(lineout[element])
    noise_axis = np.arange(0, len(noise_data))
    noise_popt, noise_pcov = so.curve_fit(exponential, noise_axis, noise_data)
    
    #fitting an exponential to the decay region to subtract away and get noise
    noise_fit = exponential(noise_axis, noise_popt[0], noise_popt[1])
    #plt.figure()
    #plt.plot(noise_axis, noise_data)
    #plt.plot(noise_axis, noise_fit)
    noise_diff = noise_data - noise_fit
    #plt.plot(noise_diff) 
    if shotnum != 0:
        plt.savefig(f'noiseFit_{shotnum}_{channel_num}.png')
        plt.close()

    # finding the mean squared noise value
    noise2 = noise_diff**2
    MSE = np.average(noise2) 
    SNR = (MSS/MSE)**.5
    SNR_inv = (MSE/MSS)**.5

    nps = 100
    smoothed_signal = nd.median_filter(signal_data, 3)
    signal_psd = welch(smoothed_signal, nperseg = nps)
    noise_psd = welch(signal_data - smoothed_signal, nperseg = nps)
    #plt.figure()
    #plt.plot(signal_psd[0], signal_psd[1])
    #plt.plot(noise_psd[0], noise_psd[1])
    #plt.legend(['signal', 'noise'])
    #plt.xlabel('freq')
    #plt.ylabel('psd')
    #plt.xlim([0, .1]) 
    SNR_freq = signal_psd[0]
    iSNR = np.sqrt(noise_psd[1]/signal_psd[1])


    print(f'Mean squared noise value: {MSE:.2f}') 
    print(f'Estimated signal to noise ratio = {SNR:.2f}')
    print(f'Estimaged inverse SNR = {SNR_inv:.2f}')

    # building a periodogram for the noise data:
    noise_freqs, noise_power = ss.periodogram(noise_diff - np.average(noise_diff)) 
    # building a periodogram for the signal:
    signal_freqs, signal_power = ss.periodogram(signal_data)
    
    # PLOTTING POWER DENSITY SPECTRUM: 
    #------------------------------------------------------------------------------------------------------------------
    '''
    fig, ax = plt.subplots(1,2)
    ax[0].scatter(noise_freqs, noise_power) 
    ax[0].set_title('Noise PSD')
    ax[1].scatter(signal_freqs, signal_power) 
    ax[1].set_title('Signal PSD') 
    ax[0].set_xlabel('frequency') 
    ax[1].set_xlabel('frequency')
    plt.close()
    '''
    # WIENER DECONVOLUTION: 
    #------------------------------------------------------------------------------------------------------------------
    # this uses the spectral information from above to perform the wiener deconvolution
    
    # the data has to be padded to the right since there is an artificial negative signal imposed otherwise
    average_time_step = np.average(np.diff(time_axis)) 
    
    padding_axis_right = np.arange(1,padding_length)
    final_value = lineout[signal_right]
    
    padding_axis_left = np.arange(1,padding_length)
    initial_value = lineout[signal_right]
    # creating the time axis and padding data to the right side of the lineout
    signal_padding_right= final_value*(np.exp(-average_time_step*padding_axis_right/fall1) + fall2_weight*np.exp(-average_time_step*padding_axis_right/fall2))
    signal_padding_left= np.zeros_like(padding_axis_left)

    time_padding_right = padding_axis_right*average_time_step
    time_padding_left = -np.flip(padding_axis_left*average_time_step)

    # appending the padding data to the right 
    signal_padded = np.hstack((signal_padding_left, lineout[:signal_right], signal_padding_right))
    signal_time_padded = np.hstack((time_padding_left + time_axis[0], time_axis[:signal_right], time_padding_right+time_axis[signal_right]))
    shift = signal_time_padded.min()
    signal_time_padded -= signal_time_padded.min()
    # creating the IRF for the PXTD detector
    IRF = pxtd_IRF(np.array(signal_time_padded))
    #IRF = nd.gaussian_filter(IRF, 50)
    decon = wiener_decon(signal_padded, IRF, SNR_inv)
    #decon = wiener_decon_freqdep(signal_padded, IRF, SNR_freq, iSNR)
    #decon = wiener_decon_freqdep(signal_data, IRF, welch(noise_data, nperseg = 40)[1])

    return signal_time_padded[padding_length:-padding_length]+shift, decon[padding_length:-padding_length]
'''    
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
    image_rot = nd.rotate(image_clean, .0297*180/3.1415)
    image_rot = nd.median_filter(image_rot, size = 3)
    
    if right_side == -1:
        right_side = image_rot.shape[1]
    
    #taking blocks of full image that will be the basis of the lineouts once averaged
    c1_lineout_block = nd.median_filter(image_rot[c1low:c1high,left_side:right_side], size = (med_filt1, med_filt2))
    
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


def lineout_to_csv(filename, x, y):
    length = len(x)
    with open(filename, 'w') as file:
        
        for element in range(length):
            file.writelines(f'{x[element]}, {y[element]}\n')
    return









            
    
