import numpy as np
import matplotlib.pyplot as plt

def get_laser_data(shotnum):

    time = []
    laser = []
    with open(f'../results/laser_data/laser_data_{shotnum}.txt', 'r') as data:
        for line in data:
            vals = line.split(' ')
            time.append(float(vals[0]))
            laser.append(float(vals[1]))

    return time, laser
