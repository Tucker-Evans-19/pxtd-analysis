
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
from scipy.signal import find_peaks
import csv
import argparse
from scipy.signal import convolve
from scipy.linalg import convolution_matrix
import pxtd_forward_model as pfor

# default variable values
dist = 3.1 # cm
diagnostic = 'PTD' # diagnostic type (ntd or ptd)
sg_len = 10 # savitsky golay parameter
sg_order = 3
runs = 60

# The relevant delays for PTD (they are approximately 1 ns apart, but not exactly)
fidu_names = ['PTD-0','PTD-1','PTD-2','PTD-3','PTD-4','PTD-5','PTD-6', 'PTD-7','PTD-8','PTD-9','PTD-10','PTD-11','PTD-12','PTD-13','PTD-14','PTD-15']
fidu_delays = ['0000','1018','2038','3089','4077','5090','6110','6928', '8180','9196','9963','0000','0000','0000','0000','00']
ptd_delay_dict = dict(zip(fidu_names, fidu_delays))

# parsing the input arguments:
parser = argparse.ArgumentParser()
parser.add_argument('shotnum')
parser.add_argument('-d', '--diagnostic', default = 'PTD')
parser.add_argument('-p', '--plotting', default = True)
args = parser.parse_args()
shotnum  = args.shotnum
plotting = args.plotting
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

# determining the time of flight corrections
Emean = []
Estd = []
tofs = []

for reaction in reactions:
    E, std = ba.ballabio_mean_std(reaction, reaction_temps[reaction])
    Emean.append(E)
    Estd.append(std)
    tofs.append(dist/td.get_pop_velocities(reaction, E))

tofs_dict = dict(zip(reactions, tofs))
E_dict = dict(zip(reactions, Emean))
Estd_dict = dict(zip(reactions, Estd))

    
# pulling in the notch correction
corrections = []
with open('ptd_notch_correction.txt', 'r') as nfile:
    for line in nfile:
        corrections.append(float(line.strip().split(',')[1]))
corrections = np.array(corrections)

#pulling in the file information 
directory = '../input/streak_data/'
files = natsorted(os.listdir(directory))
files = [file for file in files if (shotnum in file) and ('.h5' in file) and (diagnostic in file)]

if len(files) == 0:
    print('There is no data file with the specified shot number...')
elif len(files) >= 2:
    print('There is more than one data file corresponding to this shot number. I will use the first one...')
    file  = files[0]
else:
    file = files[0]

# now we have to decide which analysis routines to use:

# PTD: ----------------------------------------------------------------------
if diagnostic == 'PTD':
    # plotting the full image
    fig, ax = plt.subplots()
    td.show_pxtd_image(directory+file)

    # depending on how many channels there are, we pull out the function that we want
    if num_channels == 2:
        lineouts = td.pxtd_2ch_lineouts(directory + file, channel_width = 12)
    elif num_channels == 3:
        lineouts = td.pxtd_3ch_lineouts(directory + file, channel_width = 12)

    # pulling the background lineout
    bg_lineout = td.pxtd_lineout(directory+file, 250, channel_width = 12)
    bg_lineout = signal.savgol_filter(bg_lineout, sg_len, sg_order)
    time, centers, fid_lineout = td.get_pxtd_fid_timing(directory + file)
        
elif diagnostic == 'NTD':
    # plotting out the full image
    fig, ax = plt.subplots()
    td.show_ntd_image(directory+file)
    time, centers, fid_lineout = td.get_ntd_fid_timing(directory + file)
    # TODO add the background lineout for the ntd setup here....
   
# subtracting off the background
for lineout in lineouts:
    lineout -= bg_lineout
    lineout /= corrections

# pulling in the p510 data that has already been analyzed
with open('../results/p510_summary.txt', 'r') as p510_file:
    p510_reader = csv.DictReader(p510_file) # pulling out all of the data from the p510_file
    for line in p510_reader:
        if shotnum in line['shot number']:
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
time = time + t0_minus_fid0 - 705 + float(ptd_delay_dict[delay_box.strip(' ')])  - float(ptd_delay_dict['PTD-1']) # p510 first fiducial with teh 1 ns trim fiber in place
# TODO get the T0 from the laser in here!!!

# plotting the tim traces for each of the channels
plt.figure()
plt.plot(time, fid_lineout)

deltat = np.average(time[1:] - time[:-1])
fig, ax = plt.subplots()
ax.plot(bg_lineout, c = 'red')
for lineout in lineouts:
    ax.plot(lineout, c = 'black', linestyle = '--') 
    ax.plot(lineout - bg_lineout, c = 'black', linestyle = '-')
ax.set_title('bg subtraction')
ax.set_ylabel('lineout')
ax.set_xlabel('pixels')

# marking out the time per pixel for reference
plt.figure()
plt.plot(time, lineout)
plt.text(1,1,f'delta t: {deltat}')

# plotting the lineouts
data_array_full = np.array(lineouts)
data_array_full = data_array_full[:,50:500]
time = time[50:500]
time_full = np.array(time) 
data_array = data_array_full[:,::]
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

fig, ax = plt.subplots()
for channel in channel_data:
    ax.plot(time, channel)
ax.set_yscale('linear')

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
    variances.append(np.var(channel_noise))

IRF = td.pxtd_conv_matrix(time) # convolution matrix for the PTD and NTD scintillators (very similar though not exactly the same :) )

channel_ls_list = []
channel_averages = []
channel_stds = []
channel_peak_inds = []
channel_peak_times = []
channel_reactions = []

for channel in range(num_channels):
    channel_reactions.append([])
    channel_ls_list.append([])
    channel_peak_times.append([])
    for i in tqdm(range(runs)):
        c_noisy = channel_norms[channel] + np.random.normal(size = channel_norms[channel].size, scale = variances[channel]**.5)
        c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
        c_ls, _, _, _ = sl.lstsq(IRF, c_smooth)
        channel_ls_list[-1].append(c_ls)
    print(np.array(channel_ls_list)[channel, :,:].shape)
    channel_averages.append(np.sum(np.array(channel_ls_list)[channel, : ,:], axis = 0)/runs) # taking the average of all of runs
    channel_peak_inds.append(find_peaks(channel_averages[-1], height = 0.3 * np.max(channel_averages[-1]), distance = 50)[0]) # adding the most recent list of peaks
    print(channel_peak_inds)
    channel_peak_times[-1] = [time[int(index)] for index in channel_peak_inds[-1]] # getting the times that correspond to each of the peaks
    channel_stds.append(np.std(np.array(channel_ls_list)[channel, : ,:], axis = 0))

fig, ax = plt.subplots(num_channels,1)
for channel in range(num_channels):
    ax[channel].plot(time, channel_averages[channel], zorder = 10, color = 'maroon')
    ax[channel].fill_between(time, channel_averages[channel] - channel_stds[channel], channel_averages[channel] + channel_stds[channel], alpha = .1, zorder = 10, color = 'maroon')
    ax[channel].vlines(channel_peak_times[channel], ymin = 0, ymax = channel_averages[channel].max(), color = 'black', linestyle = '--')
    

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

# now we want to try and fit an emission history to each of the histories using the constraint that the emission should all happen within 
# the max spread of the rough bang times that we found above

# we are first going to interpolate every channels signal to a regular grid
time_res = 20
buffer = 400 # this is a buffer on the left so the the initial gaussian is not cut off

time_interp = np.linspace(0 - buffer, time.max(), int((time.max() + buffer)/time_res))
channels_interp = []
for channel in range(num_channels):
    channels_interp.append(np.interp(time_interp, time, channel_averages[channel]))

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
        spectra[-1].append(td.synth_spec_gauss(reaction, reaction_temps[reaction], num_particles = 20000)[0])
        time_trace = td.time_trace_at_dist(dist, reaction, spectra[-1][-1], birth_time = 0, time_bins = time_interp - np.min(time_interp))[1]
        time_traces[-1].append(time_trace)
        tof_matrices[-1].append(convolution_matrix(time_trace, time_interp.size, mode = 'full')[:time_interp.size, :])

if plotting:
    fig, ax = plt.subplots()
    for channel in range(num_channels):
        for trace in time_traces[channel]:
            ax.plot(time_interp[:-1], trace)
fig, ax = plt.subplots()
ax.contourf(tof_matrices[0][-1])
    
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

reaction_inds = []
reaction_signals = []
decon_noise_std = np.std(channels_interp[channel] - signal.savgol_filter(channels_interp[channel], 15, 3))
for channel in range(num_channels):
    fig, ax = plt.subplots()
    ax.plot(channels_interp[channel], color = 'k')

    reaction_inds.append([])
    reaction_signals.append([])
    for reaction in channel_reactions[channel]:
        reaction_clicks = plt.ginput(2)
        rclick_l = int(reaction_clicks[0][0])
        rclick_r = int(reaction_clicks[1][0])
        reaction_inds[-1].append([rclick_l, rclick_r])
        reaction_signal = np.copy(channels_interp[channel])
        reaction_signal[:rclick_l] *= 0
        reaction_signal[rclick_r:] *= 0
        reaction_signals[-1].append(reaction_signal)
    plt.close()

fig, ax = plt.subplots()
for channel in range(num_channels):
    for reaction_signal in reaction_signals[channel]:
        ax.plot(reaction_signal)
        
ehists = []
ehist_unc = []
tof_decons = []
for channel in range(num_channels):
    ehists.append([])
    ehist_unc.append([])
    tof_decons.append([])
    for rind, reaction in enumerate(channel_reactions[channel]):
        tof_decons[-1].append([])
        for run in tqdm(range(runs)):
            for i in tqdm(range(runs)):
                c_noisy = channel_norms[channel] + np.random.normal(size = channel_norms[channel].size, scale = variances[channel]**.5)
                c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
                c_ls, _, _, _ = sl.lstsq(IRF, c_smooth)
                try:
                    noise = np.random.normal(loc = 0 , scale = decon_noise_std, size = time_interp.size)
                    tof_decons[-1][-1].append(so.nnls(tof_matrices[channel][rind], c_ls)[0])
                except(np.linalg.LinAlgError):
                    pass
                except(RuntimeError):
                    pass
        ehists[-1].append(np.average(np.array(tof_decons[channel][rind]), axis = 0))
        ehist_unc[-1].append(np.std(np.array(tof_decons[channel][rind]), axis = 0))
    #ehists[-1].append(so.nnls(tof_matrices[channel][rind], signal.savgol_filter(reaction_signals[channel][rind], 20, 3))[0])


general_reactions = ['DTn', 'DDn', 'D3Hep']
general_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
color_dict = dict(zip(general_reactions, general_colors))


# plotting the deconvolved reaction histories
fig, ax = plt.subplots(dpi = 200)
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        ax.plot(time_interp, ehists[channel][rind], label = reaction, color = color_dict[reaction])
        ax.fill_between(time_interp,(ehists[channel][rind] - ehist_unc[channel][rind]), (ehists[channel][rind] + ehist_unc[channel][rind]), alpha = .2, color = color_dict[reaction])
ax.set_xlim([900, 1500])
ax.set_ylim([0, .0004])
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
