import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import plasma_analysis.tdstreak as td
import plasma_analysis.ballabio as ba
import scipy.special as sp
from numpy.matlib import repmat
from scipy.linalg import convolution_matrix

rm_proton = 938.272 #MeV
rm_neutron = 939.565 #MeV
c = .02998 #cm/ps

def lorentzian(t, max_val, center, width):
    return max_val * (4*((t - center)/width)**2 + 1) **-1
# from magnus falks' thesis, it is reasonable to model the evolution of the rhoR with a lorentzian function: 
# THe other reference for this is Johans 2004 PoP
def rhoR_lorentzian(t, max_rhoR, Tp, deltaT):
    # t is the time basis desired
    # max_rhoR is the rhoR at peak compression
    # Tp is the time of peak compression
    # deltaT describes the FWHM of the lorentzian
    return lorentzian(t, max_rhoR, Tp, deltaT)

def pxtd_inst(t, t0=0, dist = 5, Td3hep = 5, Tdtn = 5, Tddn = 5, Yd3hep = 1000, Ydtn = 1000, Yddn = 1000):
    pop_d3hep, mean, std = td.synth_spec_gauss('D3Hep', Td3hep, num_particles = Yd3hep, birth_time = t0)
    pop_dtn, mean, std = td.synth_spec_gauss('DTn', Tdtn, num_particles = Ydtn, birth_time = t0)
    pop_ddn, mean, std = td.synth_spec_gauss('DDn', Tddn, num_particles = Yddn, birth_time = t0)
    
    time_bins_d3hep, trace_d3hep = td.time_trace_at_dist(dist, 'D3Hep', pop_d3hep, birth_time  = t0, time_bins = t)
    _, trace_dtn = td.time_trace_at_dist(dist, 'DTn', pop_dtn, birth_time  = t0, time_bins = t)
    _, trace_ddn = td.time_trace_at_dist(dist, 'DDn', pop_ddn, birth_time  = t0, time_bins = t)

    return time_bins_d3hep, trace_d3hep, trace_dtn, trace_ddn

def pxtd_cont(t, emission_times, emission_hist, Yd3hep, Ydtn, Yddn, dist = 5, Td3hep = 5, Tdtn = 5, Tddn = 5):
    d3hep_tot = np.zeros(len(t)-1)
    dtn_tot = np.zeros(len(t)-1)
    ddn_tot =np.zeros(len(t)-1)

    

    for count, time in enumerate(emission_times):
        time_bins, trace_d3hep, trace_dtn, trace_ddn = pxtd_inst(t, dist = dist, Td3hep = Td3hep, Tdtn = Tdtn, Tddn = Tddn, Yd3hep =np.round(Yd3hep*emission_hist[count]), Ydtn = np.round(Ydtn*emission_hist[count]), Yddn = np.round(emission_hist[count]*Yddn), t0 = time)
 
        d3hep_tot += trace_d3hep
        dtn_tot += trace_dtn
        ddn_tot += trace_ddn

    return time_bins, d3hep_tot, dtn_tot, ddn_tot


def gaussian(x, A, b, C):
    gaussian = np.exp(-(x-b)**2/(2*C**2))
    gaussian *= A
    return gaussian

def skew_gaussian(x, a, b, c, d):
    shape = np.exp(-(x-b)**2/(2*c**2)) * (1/(1+np.exp(d*(x - b)/c**2)))
    shape /= shape.max() + .00001
    shape *= a 
    return shape
    
def super_gaussian(x, a, b, c, p):
    shape = np.exp(-((x-b)**2/(2*c**2))**p)
    shape /= shape.max() + .00001
    shape *= a
    return shape

def xrays_forward(t_obs, t_emission, emission, dist = 5):
    tof = dist/c
    t_emission_shifted = t_emission + tof
    obs = np.interp(t_obs, t_emission_shifted, emission)
    return obs

def xray_tof(dist = 5):
    tof = dist/c
    return tof
    
def means_and_sigmas(Td3hep, Tdtn, Tddn):
    # evaluating the mean and std of energy at every time step 
    mean_energy_d3hep, sigma_energy_d3hep = ba.ballabio_mean_std('D3Hep', Td3hep)
    mean_energy_dtn, sigma_energy_dtn = ba.ballabio_mean_std('DTn', Tdtn)
    mean_energy_ddn, sigma_energy_ddn = ba.ballabio_mean_std('DDn', Tddn)

    return mean_energy_d3hep, sigma_energy_d3hep,mean_energy_dtn, sigma_energy_dtn, mean_energy_ddn, sigma_energy_ddn

def dtn_back_basic(Tdtn,dist = 5):
    spread = np.sqrt(Tdtn) * 1.12 * dist
    Emean, Estd = ba.ballabio_mean_std('DTn', Tdtn)
    v = td.get_pop_velocities('DTn', Emean) 
    tof = dist/v
    return tof, spread

# TODO: fix the proton version please!!!!!!
def d3he_back_basic(Td3hep, Eloss = 0, dist = 5):
    spread = np.sqrt(Td3hep) * 1.12 * dist
    Emean, Estd = ba.ballabio_mean_std('D3Hep', Td3hep)
    Emean -= Eloss
    v = td.get_pop_velocities('D3Hep', Emean) 
    tof = dist/v
    return tof, spread
    
def ddn_back_basic(Tddn,dist = 5):
    spread = np.sqrt(Tddn) * 7.78 * dist
    Emean, Estd = ba.ballabio_mean_std('DDn', Tddn)
    v = td.get_pop_velocities('DDn', Emean) 
    tof = dist/v
    return tof, spread

def dE_matrix(tobs, tem, dist = 5):
    print('Creating energy matrix...')
    Tobs = repmat(tobs, len(tem), 1) # creating arrays of the time basis vectors (to do differencing calc)
    Tem = repmat(tem, len(tobs), 1)
    print(Tobs.shape)
    dt = tem[1] - tem[0] # calculating the step in the time basis
    print(f'time step: {dt}')
    deltat = Tobs - Tem.transpose() # finding array of differences
    cfilter = (deltat > dist/c)
    dv = dist*dt*deltat**-2
    dv *= cfilter
    v = dist/deltat
    v *= cfilter
    
    # getting dE, the change in energy per bin considered in time series
    dE = rm_proton*(-.5*(1-v**2/c**2)**-1.5 * (-2*v/c**2)) * dv
    beta2 = v**2/c**2
    gamma = (1-beta2)**-.5
    dE = dE.transpose()

    #getting the energies associated with each pair of time bins (obs and em)
    E = (gamma-1)*rm_proton # rest mass energy fo the proton in kev
    E = E.transpose()
    E[np.isnan(E)] = 0 
    dE[np.isnan(dE)] = 0 

    plt.figure()
    plt.imshow(dE)
    return dE, E

def pxtd_int(t, emission_times, Yd3hep, Ydtn, Yddn, mean_d3hep,sigma_d3hep, mean_dtn,sigma_dtn,mean_ddn,sigma_ddn, dE, E, dist = 5):
    # setting up dummy matrices for the total emission arriving at a detector at dist
    
    d3hep_array= dE*gaussian(E, Yd3hep, mean_d3hep, sigma_d3hep)
    dtn_array= dE*gaussian(E, Ydtn, mean_dtn, sigma_dtn)
    ddn_array= dE*gaussian(E, Yddn, mean_ddn, sigma_ddn)

    d3hep_tot = np.sum(d3hep_array, axis = 1)
    dtn_tot = np.sum(dtn_array, axis = 1)
    ddn_tot = np.sum(ddn_array, axis =1)

    return t,d3hep_tot, dtn_tot, ddn_tot

def pxtd_with_IRF(t, emission_times, Yd3hep, Ydtn, Yddn, mean_d3hep,sigma_d3hep, mean_dtn,sigma_dtn,mean_ddn,sigma_ddn, dE, E,irf_conv_matrix, dist = 5):
    # getting the observed yield vs. time at the detector from above
    t, d3hep_tot, dtn_tot, ddn_tot = pxtd_int(t, emission_times, Yd3hep, Ydtn, Yddn, mean_d3hep, sigma_d3hep, mean_dtn, sigma_dtn, mean_ddn, sigma_ddn, dE, E, dist = dist)
    # convolving with the IRF response: 
    ytot = d3hep_tot+ dtn_tot + ddn_tot
    pxtd_tot = np.multiply(irf_conv_matrix*ytot) 
    return pxtd_tot
''' 
def pxtd_int(t, emission_times, Yd3hep, Ydtn, Yddn, mean_d3hep,sigma_d3hep, mean_dtn,sigma_dtn,mean_ddn,sigma_ddn, dist = 5):
    # setting up dummy matrices for the total emission arriving at a detector at dist
    d3hep_tot = np.zeros_like(t)
    dtn_tot = np.zeros_like(t)
    ddn_tot =np.zeros_like(t)
    
    dt = emission_times[1] - emission_times[0]
    for obs_ind, obs_time in enumerate(t):
        for emission_ind, emission_time in enumerate(emission_times):
            if emission_time < obs_time:
                deltat = obs_time - emission_time
                dv = dist*dt/deltat**2
                v = dist/deltat
                #print(f'v:{v}')
                if v<c:
                    dE = rm_proton*(-.5*(1-v**2/c**2)**-1.5 * (-2*v/c**2)) * dv
                    beta = v**2/c**2
                    gamma = (1-beta)**-.5
                    E = (gamma-1)*rm_proton # rest mass energy fo the proton in kev
                    #print(E, gaussian(E, Yd3hep[emission_ind], mean_energy_d3hep[emission_ind], sigma_energy_dtn[emission_ind]))
                    d3hep_tot[obs_ind] += dE*gaussian(E, Yd3hep[emission_ind], mean_d3hep[emission_ind], sigma_d3hep[emission_ind])
                    dtn_tot[obs_ind] += dE*gaussian(E, Ydtn[emission_ind], mean_dtn[emission_ind], sigma_dtn[emission_ind])
                    ddn_tot[obs_ind] += dE*gaussian(E, Yddn[emission_ind], mean_ddn[emission_ind], sigma_ddn[emission_ind])
    return t,d3hep_tot, dtn_tot, ddn_tot
'''
def laser_xray_irf(t, rise, fall):
    A = np.exp(-t/rise)*(1-np.exp(-t/fall))
    #A = np.exp(-t**2/rise)*sp.erf(t/fall) 
    return A/np.sum(A)
    

def pxtd_laser_xrays(t_obs, t_emission, w, lpi_rise,lpi_fall,  dist, lpi_mag, p = 10):
    # using a supergaussian for the definition of the laser drive and assuming
    # that the laser induced emission from the corona looks like an exponential growth 
    # scaling the laser drive

    # the supergaussian comes from JJ ruby thesis
    # patrick adrian said he does something similar for his experiments
    # TODO compare to growth rates for parametric instabilities that drive this kind of thing (SBS and SRS)

    # w is laser halfwidth at .02 contour
    TOF = dist/c # dist in cm, c in cm/ps, TOF in ps
    C = w/(2*np.log(50)**(1/p))**.5
    laser_profile = np.exp(-((t_emission-w)**2/(2*C**2))**p)
    '''
    xray_irf = laser_xray_irf(t_emission - min(t_emission), lpi_rise, lpi_fall)
    #lpi_emission = np.convolve(xray_irf, laser_profile, mode='full')[:len(laser_profile)]

    lpi_emission = np.zeros_like(t_obs)
    for obs_ind, obs_time in enumerate(t_obs):
        for em_ind, em_time in enumerate(t_obs):
            if obs_time > em_time:
                deltat = obs_time - em_time
                irf = laser_xray_irf(deltat, lpi_rise, lpi_fall)
                lpi_emission[obs_ind] += laser_profile[em_ind] * irf
    plt.figure()
    plt.plot(lpi_emission, c = 'blue')
    plt.plot(laser_profile, c = 'red')
    #lpi emission is growth multiplied by the laser power
    '''
    lpi_emission = lpi_mag*np.exp((t_emission-w)/lpi_rise)*laser_profile
    
    laser_decay = np.zeros_like(laser_profile)
    for decay_ind, decay_element in enumerate(laser_decay):
        decayed_val = 0
        for laser_ind in range(decay_ind):
            delta_t = t_emission[decay_ind] - t_emission[laser_ind]
            decayed_val += lpi_emission[laser_ind]*np.exp(-delta_t/lpi_fall)
        laser_decay[decay_ind] = decayed_val

    #plt.figure()
    #plt.plot(laser_decay)
    lpi_emission  = lpi_mag*laser_decay
    
    # factors of TOF above are in there to shift the signal to observed time
    tout = t_emission + TOF
    lpi_obs = np.interp(t_obs, tout, lpi_emission)
    
    return tout, lpi_obs, t_emission, laser_profile

def pxtd_hotspot_xray_emission(t_obs, t_emission, Te, rho, Zi, A, dist = 5):
    # from atzeni section 10.6.3 we have bremss emission proportional to 1.76*10^17 * sqrt(Te)*Z^3*rho/A^2 [W/g]
    # A should be the weighted average of mass numbers and Z, see atzeni 

    xray_power = 1.76 * 10**17 * np.sqrt(Te) * Zi**3 * rho / A**2 #W/g
    xray_power = np.sqrt(Te) * Zi**3 * rho / A**2 #1.76*10**17 W/g

    TOF = dist/c
    tout = t_emission + TOF
    xray_obs = np.interp(t_obs, tout, xray_power)
    
    return t_obs, xray_obs

def risefall(time, risetime, decaytime):
    g = (1 - np.exp(-time/risetime))*np.exp(-time/decaytime)
    return g/g.sum()

def risefall_matrix(time, risetime, decaytime):
    T = np.array(time) # making an array out of the time for ease
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    g = risefall(Tc, risetime, decaytime)

    G = convolution_matrix(g, Tc.size, mode = 'full')
    G = G[:Tc.size, :]

    return G, g
        
def xray_tof_matrix(time, distance): # distance in cm
    T = np.array(time) # making an array out of the time for ease
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    g = np.zeros_like(T)
    TOF = 1e12*distance*.01/(2.998e8) # TOF in ps
    TOF_ind = np.argmin((Tc-TOF)**2)
    g[TOF_ind] = 1

    G = convolution_matrix(g, Tc.size, mode = 'full')
    G = G[:Tc.size, :]

    return G, g, TOF

def DTn_tof_matrix(time, distance, energy=14.03): # distance in cm
    print(energy)
    vdtn = td.get_pop_velocities('DTn', energy)
    print('DTn TOF matrix calc')
    print(vdtn)
    T = np.array(time) # making an array out of the time for ease
    print(T.shape)
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    print(Tc.shape)
    g = np.zeros_like(T)
    TOF = distance/vdtn
    print(TOF)
    TOF_ind = np.argmin((Tc-TOF)**2)
    g[TOF_ind] = 1

    G = convolution_matrix(g, Tc.size, mode = 'full')
    G = G[:Tc.size, :]

    return G, g, TOF

def DDn_tof_matrix(time, distance, energy=2.54): # distance in cm
    vddn = td.get_pop_velocities('DDn', energy)
    T = np.array(time) # making an array out of the time for ease
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    g = np.zeros_like(T)
    TOF = distance/vddn
    TOF_ind = np.argmin((Tc-TOF)**2)
    g[TOF_ind] = 1

    G = convolution_matrix(g, Tc.size, mode = 'full')
    G = G[:Tc.size, :]

    return G, g, TOF

def D3Hep_tof_matrix(time, distance, energy=14.7): # distance in cm
    vd3hep = td.get_pop_velocities('D3Hep', energy)
    T = np.array(time) # making an array out of the time for ease
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    g = np.zeros_like(T)
    TOF = distance/vd3hep
    TOF_ind = np.argmin((Tc-TOF)**2)
    g[TOF_ind] = 1

    G = convolution_matrix(g, Tc.size, mode = 'full')
    G = G[:Tc.size, :]

    return G, g, TOF

def broadening_matrix(time, sigma):
    T = np.array(time) # making an array out of the time for ease
    Tc = T - T[0] # correcting to get rid of the initial shift in time 
    dT = T[1] - T[0] # getting the timestep
    N = 8*sigma/dT # number of time steps

    Tp = np.linspace(-int(np.floor(N*dT/2)), int(np.floor(N*dT/2)), int(N)+1)
    g = gaussian(Tp, 1, 0, sigma)
    g /= g.sum()

    G = convolution_matrix(g, Tc.size, mode = 'same')
    G = G[:Tc.size, :]

    return G, g
    

    








