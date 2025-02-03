
# Author: Tucker Evans
# Name: streak_analysis_main

import scipy.linalg as sl
import numpy as np
import emcee
import tdstreak as td
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
from numpy.random import randint
from scipy.interpolate import make_smoothing_spline
plt.style.use('tableau-colorblind10')

# default variable values
dist = 9.1 # cm
diagnostic = 'PTD' # diagnostic type (ntd or ptd)
sg_len = 30 # savitsky golay parameter, 10
sg_order = 5
#downshift = .9

#time calibration delays
xray_cal_delay = 341
ptd_calib_delay = -640-653+150 + xray_cal_delay
ntd_calib_delay = -1963+363 + xray_cal_delay
#ntd_calib_delay = -1963+363+381-41

max_time_allowed = 6000
max_emission_time = 4000
max_plot_time = 4000
subtracting_background = False
reaction_ignore_list = ['DDn']
#gamma = 30 # regularization term
# -2606


### DEFINING COLORS FOR EACH CHANNEL -----------------------------------------
channel_colors = {'0': 'red', '1': 'green', '2':'blue', '3':'purple'}

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
parser.add_argument('--plotting_irf_decon', default = False)
parser.add_argument('-r', '--runs', default = 30)
parser.add_argument('-l', '--dist', default = 9.1)
parser.add_argument('--lam', default = -1)
parser.add_argument('-f','--fid_num',default = 0)
parser.add_argument('--downshift',default = 0)

args = parser.parse_args()

shotnum  = args.shotnum
lam  = float(args.lam)
plotting = args.plotting
plotting_irf_decon = args.plotting_irf_decon
runs = int(args.runs)
dist = float(args.dist)
diagnostic = args.diagnostic
fid_num = float(args.fid_num)
downshift = float(args.downshift)
print(f'Inputted shotnum: {shotnum}') 
#------------------------------------------------------------------------


# FUNCTION DEFINITIONS -------------------------------------------------
# some functions for pulling apart some of the input file values
def str2floatlist(string):
    return [float(element) for element in string.strip(' []').split(' ')]

def str2strlist(string):
    return [str(element) for element in string.strip(' []').split(' ')]

def str2multilist(string):
    lists = string.strip(' []').split(';')
    return [l.split(' ') for l in lists]
#------------------------------------------------------------------------




# SHOT LIST CHECKING ---------------------------------------------------
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
#------------------------------------------------------------------------





# ENERGY CALCULATIONS  -------------------------------------------------------

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
#------------------------------------------------------------------------





# NOTCH CORRECTION -------------------------------------------------------------------------------
# pulling in the notch correction for PTD (there is a notch in the response of the camera. scratch damage?)
corrections = []
with open('ptd_notch_correction.txt', 'r') as nfile:
    for line in nfile:
        corrections.append(float(line.strip().split(',')[1]))
corrections = np.array(corrections)
#------------------------------------------------------------------------



# STREAK DATA ------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------





# DIAGNOSTIC SPECIFIC SETUP: -----------------------------------------------------
# now we have to decide which analysis routines to use:
# -------------------------------------------------------------------
print(f'Diagnostic in use is {diagnostic}')
print('PULLING IN THE DATA>>>>')

### PTD ####
if 'PTD' in diagnostic:
    
    time_res = 30
    if lam == -1:
        lam = .001
    else:
        pass
    down_sampling = 5
    ptd_left = 40
    ptd_right = 470
    print('....Pulling data from PTD image')
    # plotting the full image
    fig, ax = plt.subplots()
    td.show_pxtd_image(directory+file, scaling = 'log') # this creates a full plot

    # depending on how many channels there are, we pull out the function that we want
    lineouts = []
    if num_channels == 2: # usually @ 3.1 cm
        print('....Two channels extracted')
        raw_lineouts = td.pxtd_2ch_lineouts(directory + file, channel_width = 20)
    elif num_channels == 3: # usually @ 9.1 cm
        print('....Three channels extracted')
        raw_lineouts = td.pxtd_3ch_lineouts(directory + file, channel_width = 20)
    
    for lineout in raw_lineouts:
        lineouts.append(lineout[ptd_left:ptd_right])


    # pulling the background lineout
    bg_channel_index = 280
    print('....Background channel extracted at channel {bg_channel_index}')
    bg_lineout_raw = td.pxtd_lineout(directory+file, bg_channel_index, channel_width = 20)[ptd_left:ptd_right]
    #bg_lineout = signal.savgol_filter(bg_lineout, sg_len, sg_order) # smoothing the background
    bg_lineout  = make_smoothing_spline(x = np.arange(len(bg_lineout_raw)), y = bg_lineout_raw, lam=lam)(np.arange(len(bg_lineout_raw)))

    # BACKGROUND ANALYSIS AND FITTING:
    # changing the left and right bounds of the background
    bg_left = 150
    bg_right = 400

    plt.figure()
    plt.plot(bg_lineout, c = 'k',label = 'bg_smoothed')
    plt.title('Background lineout')

    ### TIMING INFORMATION ######
    time, centers, fid_lineout = td.get_pxtd_fid_timing(directory + file) # pulling the timing fiducials from the image. This is nominally always at the same location in the image. 
    time = time[ptd_left:ptd_right]
    fid_lineout = fid_lineout[ptd_left:ptd_right]


### NTD ###
elif 'NTD' in diagnostic:
    time_res = 20
    if lam == -1:
        lam = 100
    else:
        pass

    down_sampling = 5
    #sg_len = 20 # savitsky golay parameter
    #sg_order = 2
    left_screen_index = 30
    right_screen_index = 1048
    print('....Pulling data from NTD image')
    # plotting out the full image
    fig, ax = plt.subplots()
    td.show_ntd_image(directory+file)
    time, centers, fid_lineout = td.get_ntd_fid_timing(directory + file)
    time  = time[left_screen_index:right_screen_index]
    fid_lineout  = fid_lineout[left_screen_index:right_screen_index]
    # TODO add the background lineout for the ntd setup here....
    if num_channels == 4:
        raw_lineouts = td.ntd_4ch_lineouts(directory + file)
        lineouts = []
        for lineout in raw_lineouts:
            lineouts.append(lineout[left_screen_index:right_screen_index])
    else:
        lineouts = [td.ntd_lineout(directory + file, channel_center = 550, channel_width = 500)[left_screen_index:right_screen_index]]
    
    print('....Background channel extracted')
    bg_lineout = td.ntd_lineout(directory+file, 950, channel_width = 30)[left_screen_index:right_screen_index]
    #bg_lineout = signal.savgol_filter(bg_lineout, sg_len, sg_order)
#-------------------------------------------------------------------------------------------------------

max_plot_ind = np.argmax(time > max_plot_time)

# BACKGROUND SUBTRACTION STEP:
# subtracting off the background
for lineout in lineouts:
    if subtracting_background:
        print('....Subtracting the background')
        print(f'........lineout size {np.size(lineout)}') 
        print(f'........bg lineout size {np.size(bg_lineout)}') 
        lineout -= bg_lineout

    else:
        print('....No background subtraction')
    if 'PTD' in diagnostic:
        print('....correcting for ptd notch')
        lineout /= corrections[ptd_left:ptd_right]

#-------------------------------------------------------------------------------------------------------


# P510 TIMING DATA --------------------------------------------------------------------------------
# pulling in the p510 data that has already been analyzed
with open('../results/p510_summary.txt', 'r') as p510_file:
    print('....P510 data extraction beginning')
    p510_reader = csv.DictReader(p510_file) # pulling out all of the data from the p510_file
    for line in p510_reader:
        if shotnum in line['shot number']:
            print('....Found the P510 file we wanted')
            t0_minus_fid0 = float(line['t0-fid0']) # getting the time between the first fiducial and the 2% rise time of the laser
# ----------------------------------------------------------------------------------------------------


# FIXING TIME AXIS ---------------------------------------------------------------------------------------
# getting the real time axis (corrected for the delays but not the tof)
print(f't0-fid0: {t0_minus_fid0}')
if diagnostic =='PTD':
    print('....time analysis for PTD')
    time = time -fid_num*548.24 - t0_minus_fid0 + float(ptd_delay_dict[delay_box.strip(' ')])-float(ptd_delay_dict['PTD-1']) + ptd_calib_delay# p510 first fiducial with teh 1 ns trim fiber in place
if diagnostic =='NTD':
    print('....time analysis for NTD')
    time = time - fid_num*548.24 -t0_minus_fid0 + float(ntd_delay_dict[delay_box.strip(' ')])-float(ntd_delay_dict['NTD-2'])+ntd_calib_delay# p510 first fiducial with teh 1 ns trim fiber in place


# TODO get the T0 from the laser in here!!!
# ----------------------------------------------------------------------------------------------------





# INITIAL TIME TRACES --------------------------------------------------------------------------------
# plotting the time traces for each of the channels
print('....plotting the time traces')


# fiducial lineout
fig, ax = plt.subplots()
ax.plot(time, fid_lineout, label = 'fiducial')

deltat = np.average(time[1:] - time[:-1])
fig, ax = plt.subplots()
ax.scatter(time, bg_lineout, c = 'maroon', alpha = .5)
for cnum, lineout in enumerate(lineouts):
    ax.plot(time, lineout, c = 'darkblue', linestyle = '-', alpha = .5, label = f'Channel {cnum}')
    ax.plot(time, lineout - bg_lineout, c = 'black', linestyle = '-', label = f'Channel {cnum}-BG')
ax.set_title('lineout - BG')
ax.set_ylabel('lineout')
ax.set_xlabel('pixels')
plt.text(1,1,f'delta t: {deltat}')
ax.legend()

# ----------------------------------------------------------------------------------------------------



# PLOTTING LOG OF THE CHANNEL DATA -------------------------------------------------------------------
# pulling each of the channels
data_array = np.array(lineouts)
channel_data = []
for channel in range(data_array.shape[0]):
    channel_data.append(data_array[channel,:])
    channel_data[-1] -= channel_data[-1].min()

# plotting the log of the data for each channel:
fig, ax = plt.subplots()
for count,channel in enumerate(channel_data):
    ax.plot(time, channel, label = f'channel {count}')
ax.set_yscale('log')
ax.set_xlabel('time (ps)')
ax.set_ylabel('log(channel counts)')
ax.set_title('Channel Data (Log.)')
ax.legend()

# ----------------------------------------------------------------------------------------------------



# PLOTTING THE CHANNEL DATA (LINEAR) -------------------------------------------------------------------
fig, ax = plt.subplots()
for count,channel in enumerate(channel_data):
    ax.plot(time, channel, label = count)
ax.set_yscale('linear')
ax.set_xlabel('time (ps)')
ax.set_ylabel('channel counts')
ax.set_title('Channel Data')
ax.legend()

# ----------------------------------------------------------------------------------------------------





# NORMALIZING CHANNELS -------------------------------------------------------------------------------
# getting the normalized versions of each channel and calculating noise spectrum:

# setting up the lists that will hold each channel's information
channel_norms = []
channel_norms_smooth = []
channel_norms_noise = []
channel_variances = []
channel_stds = []
channel_norms_average = []
channel_norms_up = []
channel_norms_down = []
channel_norms_noise_average = []

# performing the actual normalization and getting channel stds
fig, ax_norm = plt.subplots(num_channels)
for cnum, channel in enumerate(channel_data):
    # channel norm 
    cnorm = channel/channel.max()
    channel_norms.append(cnorm)

    # add a list for the channel in question so we can iterate 
    channel_norms_smooth.append([])
    channel_norms_noise.append([])

    #choosing random indices downsampled
    csize = channel_norms[0].size # the number of pixels that we're looking at
    indices = np.arange(csize) # the x axis of indices that we are using


    # going through a loop, choosing new downsampled indices to fit a smooth spline to (from channel norm)
    for i in tqdm(range(runs)):
        # generating some noise with the appropriate size and std
        fit_indices = natsorted(np.unique(randint(0, high = csize, size = int(csize/down_sampling))))

        # adding the noise to the normed signal and then performing the smoothing:
        channel_norm_smooth = make_smoothing_spline(x = indices[fit_indices], y = cnorm[fit_indices], lam = lam)(indices)
        channel_norms_smooth[-1].append(channel_norm_smooth)
    
    # getting the variance of the smooths 
    channel_variances.append(np.nanvar(np.array(channel_norms_smooth[-1]), axis = 0)) # variances of the smooth
    channel_norms_average.append(np.nanmean(np.array(channel_norms_smooth[-1]), axis = 0)) # average of the smooths
    channel_norms_noise_average.append(np.nanmean(np.array(channel_norms_noise[-1]), axis = 0)) #average noise for the channels

    # channel norm averages (std. to either side of the average)
    channel_stds.append(np.sqrt(channel_variances[-1]))
    channel_norms_down.append(channel_norms_average[-1] - channel_stds[-1])
    channel_norms_up.append(channel_norms_average[-1] + channel_stds[-1])

    # plotting the  averages of the channel norms
    ax_norm[cnum].scatter(indices, cnorm, c =channel_colors[str(cnum)], alpha = .5, s = .5)
    ax_norm[cnum].plot(indices, np.squeeze(channel_norms_average[-1]),color = channel_colors[str(cnum)], alpha = 1)
    ax_norm[cnum].fill_between(indices, channel_norms_down[-1], channel_norms_up[-1], color = channel_colors[str(cnum)], alpha = .4)
    ax_norm[cnum].set_ylim([0,1.2])


### IRF DEFINITION -----------------------------------------------------------------------------------------------------
# putting together the IRF for the scintillator:
IRF = td.pxtd_conv_matrix(time) # convolution matrix for the PTD and NTD scintillators
# -----------------------------------------------------------------------------------------------------------------------


# DECONVOLUTION OF SCINTILLATOR RESPONSE ---------------------------------------------------------------------------------

# these variables will end up holding the deconvolution of the scintillator response
channel_ls_list = []
channel_averages = []
channel_peak_inds = []
channel_peak_times = []
channel_reactions = []
channel_xi2 = []
channel_smooths = []
channel_smooth_averages = []
channel_smooth_stds = []
channel_ls_avgs = []
channel_ls_stds = []

fig, ax = plt.subplots(2, sharex = True, sharey = True)
plt.subplots_adjust(wspace=0, hspace=0)
    
indices =np.arange(time.size)
for channel in range(num_channels):

    # all of the lists for holding reactions and corresponding deconvolutions
    channel_reactions.append([])
    channel_ls_list.append([])
    channel_peak_times.append([])
    channel_smooths.append([])

    # channel length
    csize = channel_norms[0].size
    channel_norm = channel_norms[channel]
    channel_std = channel_stds[channel]

    # going through a loop to bootstrap
    for i in tqdm(range(runs)):
        y = np.random.randn(indices.size)
        Y = nd.gaussian_filter(y, 20)
        Y /= np.std(Y)
        Y *= channel_std 
        Y += channel_norms_average[channel]
        
        # randomly displacing 1 to 2 indices
        if np.random.rand() > .5:
            Y = np.hstack([np.zeros(1), Y[:-1]])
            

        # performing the actual deconvolution
        c_ls, res, _, _ = sl.lstsq(IRF, Y)
        channel_ls_list[-1].append(c_ls)

    # getting the average of the deconvolutions
    channel_ls_avg = np.nanmean(np.array(channel_ls_list[-1]), axis = 0)
    channel_ls_std = np.nanstd(np.array(channel_ls_list[-1]), axis = 0)
    channel_ls_avgs.append(channel_ls_avg)
    channel_ls_stds.append(channel_ls_std)
    
    # plotting the smooth averages and the 
    ax[0].plot(indices, channel_norms_average[channel], alpha = .8, color = channel_colors[str(channel)])
    ax[0].fill_between(indices, channel_norms_down[channel], channel_norms_up[channel], color = channel_colors[str(channel)], alpha = .5)

    ax[1].plot(indices, channel_ls_avg/channel_ls_avg.max(), alpha = .8, color = channel_colors[str(channel)])
    ax[1].fill_between(indices, (channel_ls_avg - channel_ls_std)/channel_ls_avg.max(), (channel_ls_avg + channel_ls_std)/channel_ls_avg.max(), color = channel_colors[str(channel)], alpha = .5)
    
    #ax.scatter(indices, channel_norms[channel], alpha = .2, color = channel_colors[str(channel)])
    ax[0].set_ylim([-.1,1.1])
    ax[1].set_ylim([-.1,1.1])
    ax[0].set_title('Scintillator IRF Decon.')


### TOF CORRECTIONS ####################################################################################################################################################
# Logic here: we have called out all of the reactions that might show up on each channel. We should be able to forward model each of their respective emissions         
# making use of the simplifying assumptions that (1) the temperature is the same all the way through the burn, (2) the effects of compression are minimal over
# the course of the burn, and (3) that the initial distribution is maxwellian with central energies and std. corresponding to the ballabio calculation of the first
# two moments for nuclear products at a given temperature. All of these effects are coded into the td_streak library. the broadening and time of flight can be
# captured nominally in matrix form, permitting a least squares deconvolution if noise is sufficiently reduced. A more optimal choice in the future will be to
# use MCMC or similar methods to understand the potential incurred in the fitting process. 
#########################################################################################################################################################################

# ORGANIZING CHANNEL REACTIONS ---------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------------------




# CHANNEL INTERPOLATION ---------------------------------------------------------------------------------
# we are first going to interpolate every channels signal to a regular grid

# regular time grid to interpolate to
buffer = 0 # this is a buffer on the left so the the initial gaussian is not cut off
time_interp = np.linspace(0 - buffer, time.max() + buffer, int((time.max() + buffer)/time_res))
time_interp = np.linspace(0 - buffer, max_time_allowed, int((max_time_allowed + buffer)/time_res))

# interpolation of channel signals
channels_interp = []
channels_std_interp = []
for channel in range(num_channels):
    channels_interp.append(np.interp(time_interp, time, channel_ls_avgs[channel], left = 0))
    channels_std_interp.append(np.interp(time_interp, time, channel_ls_stds[channel], left = 0))

reaction_inds = []
reaction_signals = []
reaction_stds = []
# -------------------------------------------------------------------------------------------------------





# USER INPUT FOR CHANNEL REACTION INDICES  ------------------------------------------------------------
for channel in range(num_channels):

    # plotting the channel signal 
    fig, ax = plt.subplots()
    channel_interp = np.array(channels_interp[channel])
    channel_std_interp = np.array(channels_std_interp[channel])

    ax.plot(channel_interp/channel_interp.max(), alpha = .8, color = channel_colors[str(channel)])
    #ax.fill_between(np.arange(time_interp.size), (channel_interp - channel_std_interp)/channel_interp.max(), (channel_interp + channel_std_interp)/channel_interp.max(), color = channel_colors[str(channel)], alpha = .5)

    reaction_inds.append([])
    reaction_signals.append([])
    reaction_stds.append([])
    #emission_signals.append([])
    for reaction in channel_reactions[channel]:
        reaction_clicks = plt.ginput(2)
        rclick_l = int(reaction_clicks[0][0])
        rclick_r = int(reaction_clicks[1][0])
        reaction_inds[-1].append([rclick_l, rclick_r])
        reaction_signal = np.copy(channels_interp[channel])
        reaction_std = np.copy(channels_std_interp[channel])

        # setting everything outside the range of interest to zero
        for index in range(rclick_l):
            reaction_signal[index] = 0
            reaction_std[index] = 0
        for index in range(rclick_r, reaction_signal.size):
            reaction_signal[index] = 0
            reaction_std[index] = 0
        reaction_signals[-1].append(reaction_signal)
        reaction_stds[-1].append(reaction_std)

    plt.close()


# plotting the reaction signals and uncertainties on them
fig, ax = plt.subplots()
for channel in range(num_channels):
    for count, reaction_signal in enumerate(reaction_signals[channel]):
        ax.plot(reaction_signal, label = channel, color = channel_colors[str(channel)])
        ax.fill_between(np.arange(reaction_signal.size), reaction_signal - reaction_stds[channel][count], reaction_signal + reaction_stds[channel][count], alpha = .2, color = channel_colors[str(channel)])
ax.legend()
#plt.show()

# -------------------------------------------------------------------------------------------------------









# DEFINING SPECTRA FOR DECONVOLUTION -----------------------------------------------------------------------------------------
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
            if 'D3Hep' in reaction:
                spectra[-1].append(td.synth_spec_gauss(reaction, reaction_temps[reaction], num_particles = 10000000, downshift=downshift)[0])
            else:
                spectra[-1].append(td.synth_spec_gauss(reaction, reaction_temps[reaction], num_particles = 10000000)[0])
            time_trace = td.time_trace_at_dist(dist, reaction, spectra[-1][-1], birth_time = 0, time_bins = time_interp - np.min(time_interp))[1]
            #time_trace = make_smoothing_spline(x = np.arange(time_trace.size), y = time_trace, lam =3)(np.arange(time_trace.size))
            time_traces[-1].append(time_trace/time_trace.max())
            tof_matrices[-1].append(convolution_matrix(time_trace, time_interp.size, mode = 'full')[:time_interp.size, :])
            
        else:
            time_trace = td.time_trace_at_dist(dist, reaction, np.array([1]), birth_time = 0, time_bins = time_interp - np.min(time_interp))[1]
            
            time_traces[-1].append(time_trace/time_trace.max())
            tof_matrices[-1].append(convolution_matrix(time_trace, time_interp.size, mode = 'full')[:time_interp.size, :])

# plotting spectra and time traces 
if plotting:
    fig, ax = plt.subplots()
    for channel in range(num_channels):
        for count, trace in enumerate(time_traces[channel]):
            ax.plot(time_interp[:-1], trace, label = channel_reactions[channel][count])
    plt.title('spectra time traces')

# -------------------------------------------------------------------------------------------------------






# DECONVOLVING TOF RESPONSE ----------------------------------------------------------------------------------------
max_ind = np.argmax(time_interp>max_emission_time)

ehists = []
ehist_unc = []
ehist_unc_full = []
tof_decons = []
full_decons = []

# DECONVOLUTION OF THE TOF RESPONSE: 
fig, ax = plt.subplots(num_channels)

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

        reaction_signal = reaction_signals[channel][rind]
        reaction_std = reaction_stds[channel][rind]
        reaction_len = reaction_signal.size
        for run in tqdm(range(runs)):
            try:

                added_noise = nd.gaussian_filter(np.random.randn(reaction_len), 30)
                added_noise*=reaction_std/added_noise.std()
                smoothed_signal =  reaction_signal +added_noise
                if plotting_irf_decon:
                    ax[channel].plot(smoothed_signal/10000, color = 'black', alpha = .1)
                    ax[channel].plot(reaction_signals[channel][rind], color = 'blue')
                    ax[channel].plot(reaction_stds[channel][rind], color = 'green')
                
                # REGULARIZED
                tof_decon = sl.lstsq(A, smoothed_signal)[0]
                tof_decons[-1][-1].append(tof_decon)
                ax[channel].plot(tof_decon, color = 'red', alpha = .1)

            except(np.linalg.LinAlgError):
                pass
            except(RuntimeError):
                pass
        ehists[-1].append(np.average(np.array(tof_decons[channel][rind]), axis = 0))
        ehist_unc[-1].append(np.std(np.array(tof_decons[channel][rind]), axis = 0))

general_reactions = ['DTn', 'DDn', 'D3Hep', 'xray']
general_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', 'black']
color_dict = dict(zip(general_reactions, general_colors))

# plotting the deconvolved reaction histories
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

max_time = time_interp[max_ind]
max_plot_ind = np.argmax(time_interp > max_plot_time)

print(max_ind)
for channel in range(num_channels):
    for rind, reaction in enumerate(channel_reactions[channel]):
        if reaction not in reaction_ignore_list:
            ax.plot(time_interp, ehists[channel][rind]/ehists[channel][rind][:max_plot_ind].max(), label = reaction, color = color_dict[reaction])
            ax.fill_between(time_interp,(ehists[channel][rind] - ehist_unc[channel][rind])/ehists[channel][rind][:max_plot_ind].max(), (ehists[channel][rind] + ehist_unc[channel][rind])/ehists[channel][rind][:max_plot_ind].max(), alpha = .2, color = color_dict[reaction])
            ax2.plot(time_interp, ehists[channel][rind]/ehists[channel][rind][:max_plot_ind].max(), label = reaction, color = color_dict[reaction])
            ax2.fill_between(time_interp,(ehists[channel][rind] - ehist_unc[channel][rind])/ehists[channel][rind][:max_plot_ind].max(), (ehists[channel][rind] + ehist_unc[channel][rind])/ehists[channel][rind][:max_plot_ind].max(), alpha = .2, color = color_dict[reaction])



ax.set_title(f'Deconvolution: {shotnum}')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Normalized Channel Response')
ax2.set_title(f'Deconvolution: {shotnum}')
ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Normalized Channel Response')
laser_time, laser_data = get_laser_data(int(shotnum))
ax2.plot(laser_time, laser_data/np.max(laser_data), color = 'k', linestyle = '--', alpha = .5)
ax2.set_xlim([0, max_plot_time])
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

# saving the image file:
if reaction_ignore_list ==[]:
    plt.savefig(f'../results/plots/final_deconvolution_{diagnostic}_{shotnum}.png')
else:
    rs = ''
    for reaction in reaction_ignore_list:
        rs += reaction
    plt.savefig(f'../results/plots/final_deconvolution_{diagnostic}_{shotnum}_ignored{rs}.png')



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
