
## BALLABIO CALCULATION FOR RELATIVISTIC SPECTRA

# EDITED: Aug. 4, 2023 by Tucker Evans
# Created in matlab by Neel Kabadi.

# ballabio relativistic version of the spectrum calculation for PXTD calculations
import numpy as np


def ballabio_mean_std(particle, T):

    if particle ==  'DDn':
        a1E=4.69515;
        a2E=-0.040729;
        a3E=0.47;
        a4E=0.81844;
        a1w=1.7013*10**-3;
        a2w=0.16888;
        a3w=0.49;
        a4w=7.9460*10**-4;
        E0=2.4495; #MeV
        w0=82.542; #kev^(1/2)

        #interpolation formulas for deviation from standard results
        delE=(a1E/(1+a2E*T**a3E))*T**(2/3)+a4E*T; #keV
        Emean=E0+delE*10**-3; #MeV
        delw=(a1w/(1+a2w*T**a3w))*T**(2/3)+a4w*T;
        Estdev=((w0*(1+delw)*(T)**(1/2))/(2*(2*np.log(2))**(1/2)))*10**-3; #MeV 
        
    elif particle == 'DTn':
        a1E=5.30509;
        a2E=2.4736*10**-3;
        a3E=1.84;
        a4E=1.3818;
        a1w=5.1068*10**-4;
        a2w=7.6223*10**-3;
        a3w=1.78;
        a4w=8.7691*10**-5;
        E0=14.021; #Mev
        w0=177.259; #kev^(1/2)
        
        #interpolation formulas for deviation from standard results
        delE=(a1E/(1+a2E*T**a3E))*T**(2/3)+a4E*T; #keV
        Emean=E0+delE*10**-3; #MeV
        delw=(a1w/(1+a2w*T**a3w))*T**(2/3)+a4w*T;
        Estdev=((w0*(1+delw)*(T)**(1/2))/(2*(2*np.log(2))**(1/2)))*10**-3; #MeV
    elif particle == 'D3Hep':
        # ballabio does not give a full formula for D3He. This is an approximate version accurate to ~5%. 
        E0=14.630; #Mev
        w0=180.985; #kev^(1/2)
        delE=(9*T**(2/3)+T)*10**-3; #MeV
        Estdev=((w0*(T)**(1/2))/(2*np.sqrt(2*np.log(2))))*10**-3; #MeV
        Emean=E0+delE; #MeV
    else:
        print('Not acceptable particle type')
    
    return Emean, Estdev

    
