
# Author: Tucker Evans
# Name: streak_analysis_main

import scipy.linalg as sl
import numpy as np
import emcee
import plasma_analysis.tdstreak as td
import os
import matplotlib.pyplot as plt
import pxtd_forward as pfor
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

# default variable values
dist = 3.1 # cm
diagnostic = 'PTD' # diagnostic type (ntd or ptd)
sg_len = 20 # savitsky golay parameter
sg_order = 3
runs = 30

# The relevant delays for PTD (they are approximately 1 ns apart, but not exactly)
fidu_names = ['PTD-0','PTD-1','PTD-2','PTD-3','PTD-4','PTD-5','PTD-6', 'PTD-7','PTD-8','PTD-9','PTD-10','PTD-11','PTD-12','PTD-13','PTD-14','PTD-15']
fidu_delays = ['0000','1018','2038','3089','4077','5090','6110','6928', '8180','9196','9963','0000','0000','0000','0000','00']
ptd_delay_dict = dict(zip(fidu_names, fidu_delays))

# parsing the input arguments:
parser = argparse.ArgumentParser()
parser.add_argument('shotnum')
args = parser.parse_args()
shotnum  = args.shotnum
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
# getting the real time axis (corrected for the delays)
time = time + t0_minus_fid0 + 705 + float(ptd_delay_dict[delay_box.strip(' ')])  - float(ptd_delay_dict['PTD-1']) - (tofs_dict['DTn']) # p510 first fiducial with teh 1 ns trim fiber in place
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

for channel in range(num_channels):
    channel_ls_list.append([])
    for i in tqdm(range(runs)):
        c_noisy = channel_norms[channel] + np.random.normal(size = channel_norms[channel].size, scale = variances[channel]**.5)
        c_smooth = signal.savgol_filter(c_noisy, sg_len, sg_order)
        c_ls, _, _, _ = sl.lstsq(IRF, c_smooth)
        channel_ls_list[-1].append(c_ls)
    print(np.array(channel_ls_list)[channel, :,:].shape)
    channel_averages.append(np.sum(np.array(channel_ls_list)[channel, : ,:], axis = 0)/runs) # taking the average of all of runs
    channel_stds.append(np.std(np.array(channel_ls_list)[channel, : ,:], axis = 0))
    
fig, ax = plt.subplots(num_channels,1)
for channel in range(num_channels):
    ax[channel].plot(time, channel_averages[channel], zorder = 10)
    ax[channel].fill_between(time, channel_averages[channel] - channel_stds[channel], channel_averages[channel] + channel_stds[channel], alpha = .1, zorder = 10)

if False:
    p510_time, p510_data = get_p510_data(p510_dir + p510_files[str(shot_num)])
    p510_time += t0[str(shot_num)]
    
    ax0_laser = ax[0].twinx()
    ax1_laser = ax[1].twinx()

    ax0_laser.plot(p510_time, p510_data, c = 'k', alpha = 1, zorder = 0)
    ax1_laser.plot(p510_time, p510_data, c = 'k', alpha = 1, zorder = 0) 

    ax0_laser.set_ylim(0, np.max(p510_data)*1.1)
    ax1_laser.set_ylim(0, np.max(p510_data)*1.1)
    
    for i in range(2):
        ax[i].set_xlim([-100, 1200])
        ax[i].set_ylim([0, 10])

    ax[1].set_xlabel('Time (ps)')
    ax[0].set_ylabel('Deconvolution')
    ax[1].set_ylabel('Deconvolution')

    plt.suptitle(str(shot_num))
    plt.savefig(f'ptd_decon_{shot_num}.png')
    #plt.close()

    
    # c1_average and c1_std describe the most likely history and the error on it.
    # we can now fit a gaussian to get a bang time:
    p1, prop1 = find_peaks(c1_average, height = np.max(c1_average)/5, distance = 50)
    p2, prop2 = find_peaks(c2_average, height = np.max(c2_average)/5, distance = 50)

    len1 = (c1_average).size
    len2 = c2_average.size 
    
    popts1 = []
    popts2 = []
    perrs1 = []
    perrs2 = []

    fig, ax = plt.subplots()
    ax.scatter(time, c1_average, c = 'k')
    window = 24
    half_window = int(window/2)
    sys_err = 40

    for p in p1:
        peak = int(np.round(p))
        ind_range = np.arange(peak-half_window, peak+half_window)
        popt1, pcov1 = curve_fit(pfor.gaussian, time[peak-half_window:peak+half_window], c1_average[peak-half_window:peak+half_window], p0=[5, time[peak], 200], sigma = c1_std[peak-half_window:peak+half_window])
        #popt1, pcov1 = curve_fit(pfor.gaussian, time[peak-half_window:peak+half_window], c1_average[peak-half_window:peak+half_window], p0=[5, time[peak], 200])
        popts1.append(popt1)
        print(pcov1)
        print(np.diag(pcov1))
        print(np.sqrt(np.diag(pcov1)))
        print(np.sqrt(np.diag(pcov1))[1])
        perrs1.append(np.sqrt(np.diag(pcov1))[1] + sys_err)
        ax.plot(time[peak-half_window:peak+half_window], pfor.gaussian(time[peak-half_window:peak+half_window], popt1[0], popt1[1], popt1[2]), color= 'goldenrod')
    
    fig, ax = plt.subplots()
    ax.scatter(time, c2_average, c = 'k')

    for p in p2:
        peak = int(np.round(p))
        ind_range = np.arange(peak-half_window, peak+half_window)
        popt2, pcov2 = curve_fit(pfor.gaussian, time[peak-half_window:peak+half_window], c2_average[peak-half_window:peak+half_window], p0=[5, time[peak], 200], sigma = c2_std[peak-half_window:peak+half_window])
        #popt2, pcov2 = curve_fit(pfor.gaussian, time[peak-half_window:peak+half_window], c2_average[peak-half_window:peak+half_window], p0=[5, time[peak], 200])
        popts2.append(popt2)
        print(pcov2)
        perrs2.append(np.sqrt(np.diag(pcov2))[1] + sys_err)
        
        ax.plot(time[peak-half_window:peak+half_window], pfor.gaussian(time[peak-half_window:peak+half_window], popt2[0], popt2[1], popt2[2]), color= 'goldenrod')
    popt1_array = np.array(popts1)
    popt2_array = np.array(popts2)
    perr1_array = np.array(perrs1)
    perr2_array = np.array(perrs2)
    '''
    peak_times1 = natsorted(popt1_array[:,1])
    peak_times2 = natsorted(popt2_array[:,1])
    '''
    peak_times1 = popt1_array[:,1]
    peak_times2 = popt2_array[:,1]

    # TODO : calculate the temperatures for each of the species
    
    peak_times1[0] -= tofs_dict['DTn']
    try:
        peak_times1[1] -= tofs_dict['DDn']
    except(IndexError):
        pass
    peak_times2[0] -= tofs_dict['D3Hep']

    with open('ptd_bts_mi22a_c1.txt', 'a') as ptd_file:
        line  = f'{shot_num} '
        for element in range(len(peak_times1)):
            line += str(peak_times1[element]) + ' ' + str(perr1_array[element]) + ' '

        ptd_file.writelines(line + '\n')

    print(peak_times1)
    print(peak_times2)
    #plt.close()

    


        





plt.show()


