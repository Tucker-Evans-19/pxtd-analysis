
# Author: Tucker Evans
# Name: streak_analysis_main

import scipy.linalg as sl
import numpy as np
import emcee
import plasma_analysis.tdstreak as td
import os
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import re
from scipy import signal
from natsort import natsorted
import ballabio as ba
import scipy.optimize as so
from tqdm import tqdm
from numpy.fft import fft, fftfreq, ifft, fftshift
from scipy.stats import poisson
import h5py as hp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import find_peaks
import csv
import argparse
from scipy.signal import convolve
from scipy.linalg import convolution_matrix
import pxtd_forward_model as pfor
from scipy.stats import laplace
from plot_laser_data import get_laser_data
from scipy.sparse import identity
from scipy.optimize import least_squares

# default variable values
dist = 9.1 # cm
diagnostic = 'PTD' # diagnostic type (ntd or ptd)
sg_len = 30 # savitsky golay parameter, 10
sg_order = 4
#runs = 30
ptd_calib_delay = -640
ntd_calib_delay = -2500
regularizing = True
lam = 0.001
max_time_allowed = 6600
max_emission_time = 4300
subtracting_background = False
gamma = 30 # regularization term
# -2606

### PTD and NTD Delay Boxes ---------------------------------------------
# The relevant delays for PTD (they are approximately 1 ns apart, but not exactly)
fidu_names = ['PTD-0','PTD-1','PTD-2','PTD-3','PTD-4','PTD-5','PTD-6', 'PTD-7','PTD-8','PTD-9','PTD-10','PTD-11','PTD-12','PTD-13','PTD-14','PTD-15']
fidu_delays = ['0000','1018','2038','3089','4077','5090','6110','6928', '8180','9196','9963','0000','0000','0000','0000','00']
ptd_delay_dict = dict(zip(fidu_names, fidu_delays))

fidu_names = ['NTD-0','NTD-1','NTD-2','NTD-3','NTD-4','NTD-5','NTD-6','NTD-7',
    'NTD-8','NTD-9','NTD-10','NTD-11','NTD-12','NTD-13','NTD-14','NTD-15']
fidu_delays = ['0000','1389','2367','3369','4355','5433','6431','7401',
     '8360','9424','10458', '11254', '12233', '13339', '14435', '15466']
ntd_delay_dict = dict(zip(fidu_names, fidu_delays))
#------------------------------------------------------------------------

#INPUTS ----------------------------------------------------------------
# parsing the input arguments:
parser = argparse.ArgumentParser()
parser.add_argument('shotnum')
parser.add_argument('-d', '--diagnostic', default = 'PTD')
parser.add_argument('-p', '--plotting', default = True)
parser.add_argument('-r', '--runs', default = 30)
parser.add_argument('-l', '--dist', default = 9.1)
args = parser.parse_args()
shotnum  = args.shotnum
plotting = args.plotting
runs = int(args.runs)
dist = float(args.dist)
diagnostic = args.diagnostic
print(f'Inputted shotnum: {shotnum}') 

# some functions for pulling apart some of the input file values
def str2floatlist(string):
    return [float(element) for element in string.strip(' []').split(' ')]

def str2strlist(string):
    return [str(element) for element in string.strip(' []').split(' ')]

def str2multilist(string):
    lists = string.strip(' []').split(';')
    return [l.split(' ') for l in lists]

# checking through all of the shots on the input list to see if one matches
shots_available = []
with open('../input/shot_list.csv', 'r') as shot_file:
    reader = csv.DictReader(shot_file)
    for line in reader:
        shots_available.append(line['shot number'])
        print(line)
        print(line['shot number'])
        print(line['diagnostic'])
        
        if int(line['shot number']) == int(shotnum) and diagnostic in line['diagnostic']:
            print('We found a corresponding shot!')
            reactions = str2strlist(line['reactions']) # getting all of the reactions that are relevant
            temperatures = str2floatlist(line['temperatures']) #corresponding temperatures
            reaction_temps = dict(zip(reactions, temperatures))
            dist = float(line['nosecone dist'])
            reaction_channels = str2multilist(line['reaction channels'])
            num_channels = int(line['num channels'])
            delay_box = str(line['ptd delay'])
            print(f'This analysis is for {diagnostic} at {dist} cm with {num_channels}, using delay box {delay_box} and looking at the following reactions: {reactions}')

# TOF CALCULATIONS -------------------------------------------------------
# determining the time of flight corrections
Emean = []
Estd = []
tofs = []

for reaction in reactions:
    if 'xray' not in reaction: 
        print(f'....Working on ballabio for reaction: {reaction}')
        E, std = ba.ballabio_mean_std(reaction, reaction_temps[reaction])
        Emean.append(E)
        Estd.append(std)
        tofs.append(dist/td.get_pop_velocities(reaction, E))
    else:
        Emean.append(0)
        Estd.append(0)
        tofs.append(dist/td.get_pop_velocities(reaction, 1))
# times of flight for each of the species (nominal)
tofs_dict = dict(zip(reactions, tofs))
E_dict = dict(zip(reactions, Emean))
Estd_dict = dict(zip(reactions, Estd))
    
# pulling in the notch correction for PTD (there is a notch in the response of the camera. scratch damage?)
corrections = []
with open('ptd_notch_correction.txt', 'r') as nfile:
    for line in nfile:
        corrections.append(float(line.strip().split(',')[1]))
corrections = np.array(corrections)

#pulling in the streak image information for the shot number in question
directory = '../input/streak_data/'
files = natsorted(os.listdir(directory))
files = [file for file in files if (shotnum in file) and ('.h5' in file) and (diagnostic in file or diagnostic.lower() in file)]

if len(files) == 0:
    print('There is no data file with the specified shot number...')
elif len(files) >= 2:
    print('There is more than one data file corresponding to this shot number. I will use the first one...')
    file  = files[0]
else:
    file = files[0]

# ANALYSIS ROUTINES: -----------------------------------------------------
# now we have to decide which analysis routines to use:
# PTD: -------------------------------------------------------------------
print(f'Diagnostic in use is {diagnostic}')
print('PULLING IN THE DATA>>>>')
if 'PTD' in diagnostic:
    print('....Pulling data from PTD image')
    # plotting the full image
    fig, ax = plt.subplots()
    td.show_pxtd_image(directory+file) # this creates a full plot

    # depending on how many channels there are, we pull out the function that we want
    if num_channels == 2: # usually @ 3.1 cm
        print('....Two channels extracted')
        lineouts = td.pxtd_2ch_lineouts(directory + file, channel_width = 20)
    elif num_channels == 3: # usually @ 9.1 cm
        print('....Three channels extracted')
        lineouts = td.pxtd_3ch_lineouts(directory + file, channel_width = 20)

    # pulling the background lineout
    bg_channel_index = 300
    print('....Background channel extracted at channel {bg_channel_index}')
    bg_lineout = td.pxtd_lineout(directory+file, bg_channel_index, channel_width = 20)
    bg_lineout = signal.savgol_filter(bg_lineout, sg_len, sg_order) # smoothing the background
    time, centers, fid_lineout = td.get_pxtd_fid_timing(directory + file) # pulling the timing fiducials from the image. This is nominally always at the same location in the image. 

    # BACKGROUND ANALYSIS AND FITTING:
    # changing the left and right bounds of the background
    bg_left = 115
    bg_right = 400

    plt.figure()
    plt.plot(bg_lineout, c = 'k',label = 'bg_smoothed')
    plt.plot(np.arange(bg_left, bg_right), bg_lineout[bg_left:bg_right],c = 'red', label = 'bg_smoothed')
    def bg_model(x, a, b, c, d):
        return a*x**3 + b*x**2 +c*x + d

    bg_params, bg_cov = curve_fit(bg_model, np.arange(bg_left, bg_right), bg_lineout[bg_left:bg_right])
    plt.plot(bg_model(np.arange(len(bg_lineout)), bg_params[0], bg_params[1], bg_params[2], bg_params[3]), label = 'bg_fit')
    bg_lineout = bg_model(np.arange(len(bg_lineout)), bg_params[0], bg_params[1], bg_params[2], bg_params[3]) 
        
elif 'NTD' in diagnostic:
    sg_len = 20 # savitsky golay parameter
    sg_order = 2
    print('....Pulling data from NTD image')
    # plotting out the full image
    fig, ax = plt.subplots()
    td.show_ntd_image(directory+file)
    time, centers, fid_lineout = td.get_ntd_fid_timing(directory + file)
    time  = time[30:1048]
    fid_lineout  = fid_lineout[30:1048]
    # TODO add the background lineout for the ntd setup here....
    if num_channels == 4:
        raw_lineouts = td.ntd_4ch_lineouts(directory + file)
        lineouts = []
        for lineout in raw_lineouts:
            lineouts.append(lineout[30:1048])
    else:
        lineouts = [td.ntd_lineout(directory + file, channel_center = 550, channel_width = 500)]
    
    print('....Background channel extracted')
    bg_lineout = td.ntd_lineout(directory+file, 950, channel_width = 30)[30:1048]
    bg_lineout = signal.savgol_filter(bg_lineout, sg_len, sg_order)

#plt.show()

# subtracting off the background
for lineout in lineouts:
    if subtracting_background:
        print('....Subtracting the background')
        lineout -= bg_lineout
    else:
        print('....No background subtraction')
    if 'PTD' in diagnostic:
        print('....correcting for ptd notch')
        lineout /= corrections
# pulling in the p510 data that has already been analyzed
with open('../results/p510_summary.txt', 'r') as p510_file:
    print('....P510 data extraction beginning')
    p510_reader = csv.DictReader(p510_file) # pulling out all of the data from the p510_file
    for line in p510_reader:
        if shotnum in line['shot number']:
            print('....Found the P510 file we wanted')
            t0_minus_fid0 = float(line['t0-fid0']) # getting the time between the first fiducial and the 2% rise time of the laser


'''
with open('ptd_timing_report_mi22a.txt', 'a') as time_file:
    line = f'{shot_num}'
    for center in centers:
        line += (str(center) + ', ') 
    for center in centers:
        line += (str(time[center]) + ', ') 
    line += '\n'
    time_file.writelines(line)
'''
# getting the real time axis (corrected for the delays but not the tof)
print(f't0-fid0: {t0_minus_fid0}')
if diagnostic =='PTD':
    print('....time analysis for PTD')
    time = time - t0_minus_fid0 + float(ptd_delay_dict[delay_box.strip(' ')])-float(ptd_delay_dict['PTD-1']) + ptd_calib_delay# p510 first fiducial with teh 1 ns trim fiber in place
if diagnostic =='NTD':
    print('....time analysis for NTD')
    time = time - t0_minus_fid0 + float(ntd_delay_dict[delay_box.strip(' ')]) + ntd_calib_delay# p510 first fiducial with teh 1 ns trim fiber in place
# TODO get the T0 from the laser in here!!!

# plotting the time traces for each of the channels
print('....plotting the time traces')
plt.figure()
plt.plot(time, fid_lineout)

deltat = np.average(time[1:] - time[:-1])
fig, ax = plt.subplots()
ax.plot(time, bg_lineout, c = 'red', alpha = .5)
for lineout in lineouts:
    ax.plot(time, lineout, c = 'darkblue', linestyle = '-', alpha = .5) 
    ax.plot(time, lineout - bg_lineout, c = 'black', linestyle = '-')
ax.set_title('lineout - BG')
ax.set_ylabel('lineout')
ax.set_xlabel('pixels')
plt.text(1,1,f'delta t: {deltat}')

# plotting the lineouts
data_array = np.array(lineouts)
time_full = np.array(time) 
time = time[::]

# pulling each of the channels
channel_data = []
for channel in range(data_array.shape[0]):
    channel_data.append(data_array[channel,:])
    channel_data[-1] -= channel_data[-1].min()

# plotting the data for each channel:
fig, ax = plt.subplots()
for channel in channel_data:
    ax.plot(time, channel)
ax.set_yscale('log')
ax.set_xlabel('time (ps)')
ax.set_ylabel('log(channel counts)')

fig, ax = plt.subplots()
for channel in channel_data:
    ax.plot(time, channel)
ax.set_yscale('linear')
ax.set_xlabel('time (ps)')
ax.set_ylabel('channel counts')

# getting the normalized versions of each channel:
channel_norms = []
channel_smooth = []
channel_noise = []
variances = []
for channel in channel_data:
    channel_norms.append(channel/channel.max())
    channel_smooth.append(signal.savgol_filter(channel/channel.max(), sg_len, sg_order))

for channel_num in range(num_channels):
    channel_noise.append(channel_norms[channel_num] - channel_smooth[channel_num])
    variances.append(np.var(channel_noise[-1]))

# putting together the IRF for the scintillator:
IRF = td.pxtd_conv_matrix(time) # convolution matrix for the PTD and NTD scintillators (very similar though not exactly the same :) )
#IRF = nd.gaussian_filter(td.pxtd_conv_matrix(time), 40/(2.355*(time[1] - time[0])), axes = 0) # convolution matrix for the PTD and NTD scintillators (very similar though not exactly the same :) )


# INITIAL DECONVOLUTION FOR SCINTILLATOR RESPONSE: -------------------------------------
fig, ax = plt.subplots(num_channels, sharex = True, sharey = True)
if num_channels > 1:
    for channel in range(num_channels):
        c_noisy = channel_norms[channel]
        c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
        c_ls, res, _, _ = sl.lstsq(IRF, c_smooth)
        ax[channel].plot(c_noisy, c = 'black')
        ax[channel].plot(c_smooth, c = 'red')
        ax[channel].plot(c_ls)
        ax[channel].set_title(f'smoothing, channel {channel}')
else: 
    c_noisy = channel_norms[0]
    c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
    c_ls, _, _, _ = sl.lstsq(IRF, c_smooth)
    ax.plot(c_noisy)
    ax.plot(c_smooth)
plt.show()
# these variables will end up holding the deconvolution of the scintillator response
channel_ls_list = []
channel_averages = []
channel_stds = []
channel_peak_inds = []
channel_peak_times = []
channel_reactions = []
channel_xi2 = []

fig, ax = plt.subplots()

for channel in range(num_channels):
    channel_reactions.append([])
    channel_ls_list.append([])
    channel_peak_times.append([])
    channel_xi2.append([])
    for i in tqdm(range(runs)):
        c_noisy = channel_norms[channel] + np.random.laplace(size = channel_norms[channel].size, scale = 1*variances[channel]**.5)
        c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
        c_ls, res, _, _ = sl.lstsq(IRF, c_smooth)
        #def loss(x):
        #    return np.sum((np.matmul(IRF, x) - c_smooth)**2 + gamma * np.var(c_smooth))
        #c_ls = so.minimize(loss, c_ls).x
        #c_ls, _= so.nnls(IRF, c_smooth)
        channel_ls_list[-1].append(c_ls)
        ax.plot(c_smooth, c = 'r', alpha = .2)
        ax.plot(np.matmul(IRF, c_ls), c = 'black', alpha = .2) 

    channel_averages.append(np.sum(np.array(channel_ls_list)[channel, : ,:], axis = 0)/runs) # taking the average of all of runs
    ax.plot(np.matmul(IRF, channel_averages[-1]), linestyle = 'None', marker = 'o', color = 'blue')
    #channel_averages.append(np.sum(np.array(channel_ls_list)[channel, : ,:] * np.expand_dims(np.array(channel_xi2)[channel,:]**-1 + .001, axis = 1), axis = 0)/runs) # taking the average of all of runs
    channel_peak_inds.append(find_peaks(channel_averages[-1], height = 0.3 * np.max(channel_averages[-1]), distance = 50)[0]) # adding the most recent list of peaks
    print(channel_peak_inds)
    channel_peak_times[-1] = [time[int(index)] for index in channel_peak_inds[-1]] # getting the times that correspond to each of the peaks
    channel_stds.append(np.std(np.array(channel_ls_list)[channel, : ,:], axis = 0))
    #plt.show()

# plotting the deconvolutions for the scintillator and camera responses:
        
fig, ax = plt.subplots(num_channels,1)
if num_channels >1:
    for channel in range(num_channels):
        ax[channel].plot(time, channel_norms[channel], zorder  = 10, color ='black', linestyle = '--')
        ax[channel].plot(time, channel_averages[channel], zorder = 10, color = 'maroon')
        ax[channel].fill_between(time, channel_averages[channel] - channel_stds[channel], channel_averages[channel] + channel_stds[channel], alpha = .1, zorder = 10, color = 'maroon')
        ax[channel].vlines(channel_peak_times[channel], ymin = 0, ymax = channel_averages[channel].max(), color = 'black', linestyle = '--')

        # writing the scintillator deconvolutions to a text file
        A = np.concatenate((np.expand_dims(time, 0),np.expand_dims(channel_averages[channel], 0), np.expand_dims(channel_averages[channel] - channel_stds[channel], 0), np.expand_dims(channel_averages[channel] + channel_stds[channel], 0)), axis = 0)
        np.savetxt(f'../results/csv_output/deconvolution_data_IRFonly_{shotnum}_{diagnostic}_channel_{channel}.csv', A.T)
else: 
    ax.plot(time, channel_norms[channel], zorder  = 10, color ='black', linestyle = '--')
    ax.plot(time, channel_averages[channel], zorder = 10, color = 'maroon')
    ax.fill_between(time, channel_averages[channel] - channel_stds[channel], channel_averages[channel] + channel_stds[channel], alpha = .1, zorder = 10, color = 'maroon')
    ax.vlines(channel_peak_times[channel], ymin = 0, ymax = channel_averages[channel].max(), color = 'black', linestyle = '--')

plt.savefig(f'../results/csv_output/deconvolution_plot_IRFonly_{shotnum}_{diagnostic}_channel_{channel}.png')

### TOF CORRECTIONS ####################################################################################################################################################
# Logic here: we have called out all of the reactions that might show up on each channel. We should be able to forward model each of their respective emissions         
# making use of the simplifying assumptions that (1) the temperature is the same all the way through the burn, (2) the effects of compression are minimal over
# the course of the burn, and (3) that the initial distribution is maxwellian with central energies and std. corresponding to the ballabio calculation of the first
# two moments for nuclear products at a given temperature. All of these effects are coded into the td_streak library. the broadening and time of flight can be
# captured nominally in matrix form, permitting a least squares deconvolution if noise is sufficiently reduced. A more optimal choice in the future will be to
# use MCMC or similar methods to understand the potential incurred in the fitting process. 
#########################################################################################################################################################################

max_peak_spread = 200 # the maximum likely distance between any two peak emissions

# checking which reactions are on each channel:
num_reactions = len(reactions)
for r in range(num_reactions):
    print(f'reaction number: {r}')
    for channel in range(num_channels):
        print(f'channel: {channel +1}')
        print(f'reaction_channels: {reaction_channels[r]}')
        if str(channel+1) in reaction_channels[r]:
            print(f'{channel + 1} is in {reaction_channels[r]}') 
            channel_reactions[channel].append(reactions[r])
        else:
            print(f'{channel + 1} is not in {reaction_channels[r]}') 
print(f'channel reactions: {channel_reactions}') 


# FITTING GAUSSIANS TO EACH OF THE PEAKS IN THE IMAGE:
half_window = 30
def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))
channel_params = []
for channel in range(num_channels):
    for peak_ind in channel_peak_inds[channel]:
        try:
            params, covs = curve_fit(gaussian, xdata = time[peak_ind - half_window:peak_ind + half_window], ydata = channel_averages[channel][peak_ind - half_window:peak_ind + half_window], sigma = channel_stds[channel][peak_ind - half_window:peak_ind + half_window], p0 = [5, time[peak_ind], 150], maxfev = 100000)#, p0 = [1, time[peak_ind], 150]
            print(params)
            ax[channel].plot(time, gaussian(time, params[0], params[1], params[2]), color = 'cyan')
        except(ValueError):
            pass
        
    
#plt.show()
'''
# we can find a really rough bang time if we assume that all of the peak emissions happen tof before the peak oberved signal
rough_BTs = []
for channel in range(num_channels):
    channel_tofs = [tofs_dict[r] for r in channel_reactions[channel]] # getting tof for each reaction
    ordered_tofs = natsorted(channel_tofs)
    print(f'ordered tofs: {ordered_tofs}')
    print(f'channel peak times: {channel_peak_times}')
    rough_BTs.append(np.array(channel_peak_times[channel]) - np.array(ordered_tofs)) # estimating the bang time by just assuming that we can subtract off the tof and call it good (very rough)

print(f'Rough bang times: {rough_BTs}')
rough_BT_overall = rough_BTs[0][0]
'''
# now we want to try and fit an emission history to each of the histories using the constraint that the emission should all happen within 
# the max spread of the rough bang times that we found above

# we are first going to interpolate every channels signal to a regular grid
time_res = 10
buffer = 0 # this is a buffer on the left so the the initial gaussian is not cut off
time_interp = np.linspace(0 - buffer, time.max() + buffer, int((time.max() + buffer)/time_res))
time_interp = np.linspace(0 - buffer, max_time_allowed, int((max_time_allowed + buffer)/time_res))
channels_interp = []
for channel in range(num_channels):
    channels_interp.append(np.interp(time_interp, time, channel_averages[channel], left = 0))

fig, ax = plt.subplots()
for channel in range(num_channels):
    ax.plot(time_interp, channels_interp[channel])
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Channel deconvolution (ps)')

#given the specified reaction temperatures, we can define a green's function corresponding to emission during a single time step
spectra = []
time_traces = []
tof_matrices = []
for channel in range(num_channels):
    spectra.append([])
    time_traces.append([])
    tof_matrices.append([])
    for reaction in channel_reactions[channel]:
        if 'xray' not in reaction:
            spectra[-1].append(td.synth_spec_gauss(reaction, reaction_temps[reaction], num_particles = 20000)[0])
            time_trace = td.time_trace_at_dist(dist, reaction, spectra[-1][-1], birth_time = 0, time_bins = time_interp - np.min(time_interp))[1]
        else:
            time_trace = td.time_trace_at_dist(dist, reaction, np.array([1]), birth_time = 0, time_bins = time_interp - np.min(time_interp))[1]
            
        time_traces[-1].append(time_trace)
        tof_matrices[-1].append(convolution_matrix(time_trace, time_interp.size, mode = 'full')[:time_interp.size, :])

if plotting:
    fig, ax = plt.subplots()
    for channel in range(num_channels):
        for trace in time_traces[channel]:
            ax.plot(time_interp[:-1], trace)
#fig, ax = plt.subplots()
#ax.pcolormesh(tof_matrices[1][-1])
#ax.set_title('tof_matrix')
    
# we generate some reasonable guesses at the initial emission histories:
def gaussian(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

emission_histories = []
for channel in range(num_channels):
    emission_histories.append([])
    for rind, reaction in enumerate(channel_reactions[channel]):
        emission_histories[-1].append(gaussian(time_interp, 100, 1000, 100))

fig, ax = plt.subplots()
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        ax.plot(time_interp, emission_histories[channel][rind], label = reaction, linestyle = '--')
        ax.plot(time_interp, np.matmul( tof_matrices[channel][rind], emission_histories[channel][rind]),label = reaction)
    ax.legend()
'''j
sg_len = 30 # savitsky golay parameter, 10 
sg_order = 2
runs = 20
'''
reaction_inds = []
reaction_signals = []
decon_noise = (channels_interp[channel] - signal.savgol_filter(channels_interp[channel], sg_len, sg_order))
decon_noise_std = (np.std(channels_interp[channel] - signal.savgol_filter(channels_interp[channel], sg_len, sg_order)))
smoothed_noise_std = signal.savgol_filter(np.abs(decon_noise), 100, 2)
noise_fig, noise_ax = plt.subplots()
noise_ax.plot(decon_noise)
noise_ax.plot(decon_noise**2)
noise_ax.plot(smoothed_noise_std)
noise_ax.set_title('noise')
noise_ax.set_xlabel('bin')
noise_ax.set_ylabel('noise')
noise_ax.hlines([decon_noise_std],xmin = 0, xmax = decon_noise.size, color = 'k', linestyle = '--')
noise_ax.hlines([np.std(np.abs(decon_noise))],xmin = 0, xmax = decon_noise.size, color = 'red', linestyle = '--')

fig, ax = plt.subplots()
n0, bins, _ = ax.hist(decon_noise, bins = 30, alpha = .5, label = 'noise')
n, bins, _ = ax.hist(decon_noise**2, bins = 30, alpha = .5, label = 'squared noise')
ax.set_title('noise')
ax.set_xlabel('bin')
ax.set_ylabel('noise')
ax.vlines([0, decon_noise_std, -decon_noise_std],ymin = 0, ymax = n.max(), color = 'k', linestyle = '--')
bin_axis = np.linspace(-bins.max(), bins.max())
ax.plot(bin_axis, n0.max() * np.exp(-(bin_axis- np.average(decon_noise))**2/(2*decon_noise_std**2)), label = 'gaussian fit')
ax.plot(bin_axis, laplace.pdf(bin_axis, loc = 0 , scale = decon_noise_std)*n0.max()*.5, label = 'laplace fit')
ax.legend()

for channel in range(num_channels):
    fig, ax = plt.subplots()
    ax.plot(channels_interp[channel], color = 'k')

    reaction_inds.append([])
    reaction_signals.append([])
    #emission_signals.append([])
    for reaction in channel_reactions[channel]:
        reaction_clicks = plt.ginput(2)
        rclick_l = int(reaction_clicks[0][0])
        rclick_r = int(reaction_clicks[1][0])
        reaction_inds[-1].append([rclick_l, rclick_r])
        reaction_signal = np.copy(channels_interp[channel])
        for index in range(rclick_l):
            reaction_signal[index] = 0
        for index in range(rclick_r, reaction_signal.size):
            reaction_signal[index] = 0
        reaction_signals[-1].append(reaction_signal)
    plt.close()

fig, ax = plt.subplots()
for channel in range(num_channels):
    for reaction_signal in reaction_signals[channel]:
        ax.plot(reaction_signal, label = channel)
ax.legend()
       
#plt.figure()
#decon_mat = np.matmul(pfor.broadening_matrix(time_interp, 40/2.355)[0], tof_matrices[0][0])
#decon_mat = nd.gaussian_filter(tof_matrices[0][0],20/(2.355*time_res), axes = 0)
#decon_mat = tof_matrices[0][0]
#plt.pcolor(decon_mat)
#plt.figure()
#plt.pcolor(tof_matrices[0][0])

# DECONVOLVING TOF RRESPONSE ---------------------------------------------------------
max_ind = np.argmax(time_interp>max_emission_time)

ehists = []
ehist_unc = []
#ehists_full = []
ehist_unc_full = []
tof_decons = []
full_decons = []

'''
for channel in range(num_channels):
    ehists.append([])
    ehist_unc.append([])
    tof_decons.append([])
    full_decons.append([])
    for rind, reaction in enumerate(channel_reactions[channel]):
        tof_decons[-1].append([])
        full_decons[-1].append([])
        #decon_mat = nd.gaussian_filter(tof_matrices[channel][rind], 40/(2.355*time_res), axes = 0)
        for run in tqdm(range(runs)):
            try:
                #noise = np.random.normal(loc = 0 , scale = decon_noise_std, size = time_interp.size)
                #noise = np.random.normal(loc = 0 , scale = 1, size = time_interp.size)*smoothed_noise_std
                noise = np.random.normal(loc = 0 , scale = 1, size = time_interp.size)*4*smoothed_noise_std
                smoothed_signal = signal.savgol_filter(reaction_signals[channel][rind] + noise, sg_len, sg_order)
                #decon_mat = np.matmul(pfor.broadening_matrix(time_interp, 40/2.355)[0], tof_matrices[channel][rind])
                tof_decons[-1][-1].append(so.nnls(tof_matrices[channel][rind], smoothed_signal)[0])
                #tof_decons[-1][-1].append(sl.lstsq(tof_matrices[channel][rind], smoothed_signal)[0])
            except(np.linalg.LinAlgError):
                pass
            except(RuntimeError):
                pass
        ehists[-1].append(np.average(np.array(tof_decons[channel][rind]), axis = 0))
        ehist_unc[-1].append(np.std(np.array(tof_decons[channel][rind]), axis = 0))
        #ehists_full[-1].append(np.average(np.array(full_decons[channel][rind]), axis = 0))
        #ehist_unc_full[-1].append(np.std(np.array(full_decons[channel][rind]), axis = 0))
    #ehists[-1].append(so.nnls(tof_matrices[channel][rind], signal.savgol_filter(reaction_signals[channel][rind], 20, 3))[0])
'''
# DECONVOLUTION OF THE TOF RESPONSE: 

for channel in range(num_channels):
    ehists.append([])
    ehist_unc.append([])
    tof_decons.append([])
    full_decons.append([])
    for rind, reaction in enumerate(channel_reactions[channel]):
        tof_decons[-1].append([])
        full_decons[-1].append([])
        #decon_mat = nd.gaussian_filter(tof_matrices[channel][rind], 40/(2.355*time_res), axes = 0)
        reaction_filter = (reaction_signals[channel][rind]>0)
        A = tof_matrices[channel][rind]
        for run in tqdm(range(runs)):
            try:
                #noise = np.random.normal(loc = 0 , scale = decon_noise_std, size = time_interp.size)
                #noise = np.random.normal(loc = 0 , scale = 1, size = time_interp.size)*smoothed_noise_std

                # ADDING NOISE THAT MATCHES NOISE IN SIGNAL (4*SIGMA)
                noise = np.random.laplace(loc = 0 , scale = 1, size = time_interp.size)*4*smoothed_noise_std * reaction_filter
                smoothed_signal = signal.savgol_filter(reaction_signals[channel][rind] + noise, sg_len, sg_order)
                #decon_mat = np.matmul(pfor.broadening_matrix(time_interp, 40/2.355)[0], tof_matrices[channel][rind])
                #tof_decons[-1][-1].append(so.nnls(tof_matrices[channel][rind], smoothed_signal)[0])

                # STANDARD
                #tof_decons[-1][-1].append(sl.lstsq(tof_matrices[channel][rind], smoothed_signal)[0])

                # REGULARIZED
                tof_decons[-1][-1].append(sl.lstsq(A, smoothed_signal)[0])

                # LEAST SQUARES REGULARIZED
                def l2_regularized(x):
                    return np.sum(np.abs(np.matmul(A,x).T - smoothed_signal) + np.var(x)*gamma)
                #x0 = sl.lstsq(tof_matrices[channel][rind], smoothed_signal)[0]
                #x0 = np.zeros_like(smoothed_signal)
                #tof_decons[-1][-1].append(so.minimize(l2_regularized, x0).x)
            except(np.linalg.LinAlgError):
                pass
            except(RuntimeError):
                pass
        ehists[-1].append(np.average(np.array(tof_decons[channel][rind]), axis = 0))
        ehist_unc[-1].append(np.std(np.array(tof_decons[channel][rind]), axis = 0))
        #ehists_full[-1].append(np.average(np.array(full_decons[channel][rind]), axis = 0))
        #ehist_unc_full[-1].append(np.std(np.array(full_decons[channel][rind]), axis = 0))
    #ehists[-1].append(so.nnls(tof_matrices[channel][rind], signal.savgol_filter(reaction_signals[channel][rind], 20, 3))[0])


general_reactions = ['DTn', 'DDn', 'D3Hep', 'xray']
general_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', 'black']
color_dict = dict(zip(general_reactions, general_colors))

reaction_ignore_list = []
# plotting the deconvolved reaction histories
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

max_time = time_interp[max_ind]

print(max_ind)
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        if reaction not in reaction_ignore_list:
            ax.plot(time_interp, ehists[channel][rind], label = reaction, color = color_dict[reaction])
            ax.fill_between(time_interp,(ehists[channel][rind] - ehist_unc[channel][rind]), (ehists[channel][rind] + ehist_unc[channel][rind]), alpha = .2, color = color_dict[reaction])
            ax2.plot(time_interp, ehists[channel][rind]/ehists[channel][rind][:max_ind].max(), label = reaction, color = color_dict[reaction])
            ax2.fill_between(time_interp,(ehists[channel][rind] - ehist_unc[channel][rind])/ehists[channel][rind][:max_ind].max(), (ehists[channel][rind] + ehist_unc[channel][rind])/ehists[channel][rind][:max_ind].max(), alpha = .2, color = color_dict[reaction])
            #ax2.plot(time_interp, ehists_full[channel][rind], label = reaction, color = color_dict[reaction])
            #ax2.fill_between(time_interp,(ehists_full[channel][rind] - ehist_unc_full[channel][rind]), (ehists_full[channel][rind] + ehist_unc_full[channel][rind]), alpha = .2, color = color_dict[reaction])
ax.set_title(f'Deconvolution: {shotnum}')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Normalized Channel Response')
ax2.set_title(f'Deconvolution: {shotnum}')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Normalized Channel Response')
laser_time, laser_data = get_laser_data(int(shotnum))
ax2.plot(laser_time, laser_data/np.max(laser_data), color = 'k', linestyle = '--', alpha = .5)
ax2.set_xlim([0, max_time])
ax2.set_ylim([0, 1.1])

# writing out all of the deconvolutions to their own file:
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        A = np.concatenate((np.expand_dims(time_interp, 0),np.expand_dims(ehists[channel][rind], 0), np.expand_dims(ehists[channel][rind] - ehist_unc[channel][rind], 0), np.expand_dims(ehists[channel][rind] + ehist_unc[channel][rind], 0)), axis = 0)
        np.savetxt(f'../results/csv_output/deconvolution_data_{shotnum}_{diagnostic}_channel_{channel}_{reaction}.csv', A.T)
    
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        A = np.concatenate((np.expand_dims(time_interp, 0),np.expand_dims(ehists[channel][rind], 0), np.expand_dims(ehists[channel][rind] - ehist_unc[channel][rind], 0), np.expand_dims(ehists[channel][rind] + ehist_unc[channel][rind], 0)), axis = 0)
        np.savetxt(f'../results/csv_output/deconvolution_data_{shotnum}_{diagnostic}_channel_{channel}_{reaction}.csv', A.T)
 
plt.savefig(f'../results/plots/final_deconvolution_{diagnostic}_{shotnum}.png')



'''
def channel_response_score(em_histories):
    error = 0
    for channel in range(num_channels):
        s = np.zeros_like(time_interp)
        for rind, reaction in enumerate(channel_reactions[channel]):
            s += np.matmul(em_histories[channel][rind], tof_matrices[channel][rind])
        error += np.sum((channels_interp[channel] - s)**2)
    return error

'''


plt.show()
