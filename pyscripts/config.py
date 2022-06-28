# Configuration file by Bart Ouwehand 08-06-2022

import numpy as np
import matplotlib.pyplot as plt

import h5py
import scipy.signal
import logging
import time
import multiprocess
import os
import psutil

from lisagwresponse import GalacticBinary
from lisainstrument import Instrument
#from lisaorbits import KeplerianOrbits

from pytdi import Data
from pytdi import michelson as mich
from pytdi import ortho
import pytdi

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import BarycentricMeanEcliptic

from scipy.interpolate import interp1d
from tqdm import tqdm


# When using ALICE, the paths change a little
use_ALICE = False
alice_test = True

if use_ALICE:
    orbit_path = '../esa-orbits.h5'
    ext = '../../data1/'
    ext2 = '/scratchdata/s2001713/'

else:
    orbit_path = '../../orbits/esa-orbits.h5'
    ext = ''
    ext2 = 'gws_spam/'


gw_path = 'gws.h5'

# Setup simluation parameters
# fs = 0.1    # Hz
fs = 0.05    # Hz
day = 86400 # s


if alice_test:
    dur_range = np.array([5])
    alpha = np.array([1])
    N1, N2 = 1,31
else:
    dur_range = np.array([5])
    # dur_range = np.array([10,30,90,180])
    # Defining alpha to iterate over
    # alpha = np.array([1,10,100])#,10,100,1000])
    alpha = np.array([1,10,100,1000,10000])
    N1,N2 = 10, 11



duration = day*dur_range # X days
size = duration*fs
discard = 300
rec = ['A','E','T']

# Define the orbit, sample 10x a day and let it extend 110% of the duration of the simulation
# orbits_t0 = 0
# orbits_dt = day/10
# orbits_s = duration[-1]*11
# orbit = KeplerianOrbits(size=orbits_s,dt=orbits_dt, t0=orbits_t0)
with h5py.File(orbit_path) as orbits:
    orbits_t0 = orbits.attrs['t0']
    orbit_fs = 1/orbits.attrs['dt']
    orbit_dur = orbits.attrs['tsize']/orbit_fs
    print ("orbits fs = "+str(fs)+" Hz, orbits t0 = "+str(orbits_t0)+" s,  orbit_duration = "+str(orbit_dur/day)+" d")

# Turn on/off binary creation & instrument simulation
use_verbinaries = True
run_new_simulation = True
gen_plots = False
calculate_again = True

# Specify specific number of binaries & import their parameters
Ngalbins = 1
amplitude_amplification = 1e8

if use_verbinaries:
    # Define name of simulation uitput file
    sample_outputf = ext+'measurements/sampdat_'+str(duration[-1]//day)+'d'+'_verbins' #extention of .h5 or .txt added later
    
    rawdata = ascii.read("../verbinaries_data_wsource_name.txt")
    
    params = ['lGal', 'bGal', 'orbital_period', 'm1', 'm1e', 'm2', 'm2e', 'i', 'freq', 'par','epar', 'dis', 'edis', 'A', 'eA', 'SNR', 'eSNR']
    # units: lGal [deg], bGal [deg], orbital_period [s], m1 [Msol], m1e [Msol], m2 [Msol], m2e [Msol]
    # i [deg], freq (of gws) [mHz], par [mas], epar [mas], dis [pc], edis [pc], A [1e-23], eA [1e-23], SNR, eSNR
    
    sourcenames = np.array(rawdata["source"])[:Ngalbins]
    Amp_true = (np.array(rawdata["A"])* (1e-23 * amplitude_amplification))[:Ngalbins] # 10yokto to 1e-23 
    f_true = (np.array(rawdata["freq"])* (1e-3))[:Ngalbins] # mHz to Hz
    iota = np.deg2rad(np.array(rawdata["i"]))[:Ngalbins] # deg to rad
    
    # Galactic coordinates of verification binaries   
    source_gal_lon = np.array(rawdata["lGal"])[:Ngalbins]  # degree range from [0,360]
    source_gal_lat = np.array(rawdata["bGal"])[:Ngalbins]  # degree range from [-90,90]

    # Transform coordinates to (barycentric mean) ecliptic coordinates
    gc = SkyCoord(l=source_gal_lon*u.degree, b=source_gal_lat*u.degree, frame='galactic')
    gw_beta_true = np.deg2rad(gc.barycentricmeanecliptic.lon.value)[:Ngalbins] # degree to rad range [0,2pi]
    gw_lambda_true = np.deg2rad(gc.barycentricmeanecliptic.lat.value)[:Ngalbins] # degree to rad range [-pi/2,pi/2]

    # Transform coordinates to equatoral (ICRS) coordinates
    # ra = gc.icrs.ra.value # degree range [0,360]
    # dec = gc.icrs.dec.value # degree range [-90,90]
    
    totNgalbins = len(sourcenames)
    phi0_true_forinst = np.zeros(Ngalbins)
    phi0_true = np.array(ascii.read("../verbinaries_phaseoffset_"+str(int(1/fs))+"dt.txt")['phi0'])[:Ngalbins]
    print ("Number of Verification Binaries = {}".format(Ngalbins))

else:
    # Define name of simulation uitput file
    sample_outputf = 'measurements/MCMCsample'+str(int(duration))+'s' #extention of .h5 or .txt added later
    
    totNgalbins = 2
    Amp_true = np.array([1e-16,5e-13])[:Ngalbins]
    f_true = np.array([1e-3,1e-4])[:Ngalbins]
    
    phi0_true_forinst = np.zeros(Ngalbins)
    phi0_true = np.array([-0.4,0.2])[:Ngalbins]
    gw_beta_true = np.array([0,0])[:Ngalbins]
    gw_lambda_true = np.array([0,np.pi])[:Ngalbins]

# Amp_range = (Amp_true * (np.array((list(np.linspace(0.1,2.5,N2))*Ngalbins)).reshape(Ngalbins,N2)).T).T
# Amp_range = (Amp_true * (np.array((list(np.linspace(0.1,1.9,N2))*Ngalbins)).reshape(Ngalbins,N2)).T).T
Amp_range = (Amp_true * (np.array((list(np.logspace(-3,3,N2))*Ngalbins)).reshape(Ngalbins,N2)).T).T
