import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import re
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser() 
parser.add_argument('shotnum') # positional argument to set the shot number
parser.add_argument('-p', '--plot', help = 'True or False do you want to plot?')

args = parser.parse_args()
print(args)

print(args.shotnum)

shotnum = args.shotnum

p510_files = os.listdir('../input/p510_data')
shots_available = []
for file in p510_files:
    try:
        shots_available.append(re.findall(r'(\d{6})', file)[0])
    except(IndexError):
        try:
            shots_available.append(re.findall(r'(\d{5})', file)[0])
        except(IndexError):
            pass
        pass
shots_available = np.unique(shots_available)

print(shots_available)

if not (str(shotnum) in shots_available):
    print('shot not available')
else:
    print('we got that shot in the p510 data')
    files = [file for file in p510_files if str(shotnum) in file]
    file = files['_2' in files]
    file = '../input/p510_data/' + file
    print(f'Analyzing {file}')
    data_file = h5py.File(file)
    streak_image = data_file['Streak_array'][0]
    fig, ax = plt.subplots(2)
    contour_plot = ax[0].contourf(streak_image)
    laser_strip = np.sum(streak_image[58:465,: ], axis = 0)
    laser_strip = laser_strip - np.average(laser_strip[400:500])
    fid_strip = np.sum(streak_image[480:500, :], axis = 0)
    plt.colorbar(contour_plot)
    ax[1].plot(laser_strip.T/laser_strip.max(), c = 'green')
    ax[1].twinx().plot(fid_strip/fid_strip.max(), c = 'blue')
    fid_peaks = find_peaks(fid_strip, height = np.max(fid_strip)*.7, distance = 30)[0]

    def quadratic_time(x, a, b, c):
        return a*x**2 + b*x +c
    
    num_peaks = len(fid_peaks)
    peak_times = (np.arange(num_peaks)-1)*548.24
    popt, pcov = curve_fit(quadratic_time, fid_peaks, peak_times)
    print(popt)

    bins = np.arange(len(laser_strip))
    time = quadratic_time(bins, popt[0], popt[1], popt[2]) 


    laser_ind0 = np.argmax(laser_strip > .02*laser_strip.max())
    ax[1].vlines(fid_peaks, ymin = 0, ymax = 1, color = 'black')
    ax[1].vlines(laser_ind0, ymin = 0, ymax = 1, color = 'r')

    average_fid_dist = np.average(fid_peaks[1:] - fid_peaks[:-1])
    print(fid_peaks)
    print(laser_ind0)

    etalon = 548.24 #ps
    ps_per_pixel = etalon/average_fid_dist

    # output time axis
    time = np.arange(laser_strip.size, dtype = 'float')
    time -= laser_ind0
    time *= ps_per_pixel

    # output laser data to file:
    A = np.concatenate((np.expand_dims(time, 0), np.expand_dims(laser_strip, 0)))
    np.savetxt(f'../results/laser_data/laser_data_{shotnum}.txt', A.T)
    
    

    line = ''
    lines = []

    with open('../results/p510_summary.txt', 'r') as summary_file:
        lines = [line for line in summary_file if shotnum not in line]
    with open('../results/p510_summary.txt', 'w') as summary_file:
        for line in lines:
            summary_file.writelines(line)
        new_line = f'{shotnum}, {fid_peaks}, {laser_ind0 * ps_per_pixel}, {ps_per_pixel}, {(laser_ind0 - fid_peaks[0]) * ps_per_pixel}\n'
        print('shotnum, fid peak times, laser t0 value, ps_per_pixel, laser_time - fid_time')
        print(new_line)
        summary_file.writelines(new_line)

plt.show()

