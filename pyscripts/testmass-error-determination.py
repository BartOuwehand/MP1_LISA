# By Bart Ouwehand 12-04-2022
import numpy as np
import matplotlib.pyplot as plt

import h5py
import scipy.signal
import logging
import time
import multiprocess
import os

from lisagwresponse import GalacticBinary
from lisainstrument import Instrument

from pytdi import Data
from pytdi import michelson as mich
from pytdi import ortho

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import BarycentricMeanEcliptic

# Import own functions
from my_functions import *


orbit_path = '../../orbits/keplerian_long.h5'
gw_path = 'gws.h5'

# Setup simluation parameters
fs = 0.1    # Hz
day = 86400 # s
duration = day*5 # X days
size = duration*fs
discard = 300
rec = ['A','E','T']

with h5py.File(orbit_path) as orbits:
    orbits_t0 = orbits.attrs['t0']
    orbit_fs = 1/orbits.attrs['dt']
    orbit_dur = orbits.attrs['tsize']/orbit_fs
    print ("fs = "+str(fs)+" Hz,  orbit_duration = "+str(orbit_dur/day)+" d")

# Turn on/off binary creation & instrument simulation
use_verbinaries = True
run_new_simulation = False
gen_plots = True

# Specify specific number of binaries
Ngalbins = 16

# Insert binary parameters
amplitude_amplification = 1e4

if use_verbinaries:
    # Define name of simulation uitput file
    sample_outputf = 'measurements/MCMCsample'+str(int(duration))+'s'+'_verbins' #extention of .h5 or .txt added later
    
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
    phi0_true = np.array(ascii.read("../verbinaries_phaseoffset.txt")['phi0'])[:Ngalbins]
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


if run_new_simulation:
    # Generate the Galactic binaries
    GenerateGalbins(orbit_path,gw_path,fs,size,Amp_true, f_true, phi0_true_forinst, gw_beta_true, gw_lambda_true, orbits_t0 + 10)
    
    # Generate the instrument data
    sAfunc, sEfunc, sTfunc = GenerateInstrumentAET(orbit_path, gw_path, fs, size, sample_outputf, discard)

# Retreive A, E, T data
rawdata = ascii.read(sample_outputf+'.txt')
sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E'],rawdata['T']])

# Plot raw data retreived from AET datastream
if gen_plots:
    for i in range(3):
        plt.plot(sdata[0]/day,sdata[i+1],label=rec[i],alpha=.5)
    plt.title('AET datastreams for '+str(duration//day)+'d simulation with all noises')
    plt.legend(loc=4)
    plt.xlabel('Time (d)')
    plt.ylabel('Amplitude')
    plt.savefig("plots/Sample_rawAETdata.jpg")

def psd_func(data):
    return scipy.signal.welch(data,fs=fs,window='nuttall',nperseg=len(data),detrend=False)

# Create psd for data
tmp = []
for i in range(1,4):
    ftmp, psdtmp = psd_func(sdata[i])
    tmp.append(psdtmp)
psd = np.array([ftmp,tmp[0],tmp[1],tmp[2]])

# Create filtered data
cutoff = 100
tmp = []
#coeffs = scipy.signal.firls(73,bands=[0,1,1.2,2],desired=[1,1,0,0],fs=fs)
#coeffs = scipy.signal.firls(73, bands=[0,1e-2,3e-2,5e-2], desired=[1,1,0,0],fs=fs)
coeffs = scipy.signal.firls(73, bands=[0,1e-2,2e-2,fs/2], desired=[1,1,0,0],fs=fs)
for i in range(1,4):
    fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=sdata[i],padlen=len(psd[0]))
    tmp.append(fdata_tmp[cutoff:-cutoff])
fsdata = np.array([sdata[0][cutoff:-cutoff],tmp[0],tmp[1],tmp[2]])

# Create psd for filtered data
tmp = []
for i in range(1,4):
    ftmp, psdtmp = psd_func(fsdata[i])
    tmp.append(psdtmp)
fpsd = np.array([ftmp,tmp[0],tmp[1],tmp[2]])

if gen_plots:
    fig, axs = plt.subplots(3, figsize=(16,6), sharex=True, gridspec_kw={'hspace':0})
    fig.suptitle("Filtered datastreams for a "+str(duration//day)+" day simulation")
    for i in range(3):
        axs[i].plot(fsdata[0]/day,fsdata[i+1],label='Channel '+rec[i])
        axs[i].legend(loc=1)
        axs[i].set_ylabel('Amplitude')
    axs[i].set_xlabel('Time (d)')
    axs[i].set_xlim(.25,.75)
    plt.savefig('plots/Sample_filteredAETdata.jpg')

    fig, axs = plt.subplots(3, figsize=(24,9), sharex=True, gridspec_kw={'hspace':0})
    for i in range(3):
        for f in f_true:
            if f == f_true[0]:
                axs[i].plot([f]*2,[1e-50,1e50],c='black',alpha=0.5,label="verbins")
            else:
                axs[i].plot([f]*2,[1e-50,1e50],c='black',alpha=0.5)
        axs[i].plot(psd[0],psd[i+1], label=rec[i]+' unfiltered',c='blue')
        axs[i].plot(fpsd[0],fpsd[i+1], label=rec[i]+' filtered',c='red')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')
        axs[i].set_xlabel('Freq [Hz]')
        axs[i].set_ylabel('ASD [Hz/sqrt(Hz)]?')
        axs[i].set_ylim(ymin=0.1*np.min(fpsd[i+1]),ymax=10*np.max(psd[i+1]))
        # axs[i].set_title("PSD of chanel "+rec[i]+' filtered and unfiltered')
        axs[i].grid()
    axs[0].set_title("PSD of chanels filtered and unfiltered")
    axs[i].set_xlim(xmin=9e-5,xmax=fs/1.9)#,xmax=1e-2)
    plt.savefig("plots/Sample_PSD.jpg")

# Build the TDI chanels for model data
Afunc, Efunc = BuildModelTDI(orbit_path,fs,size,Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=False)


# Defining the model used for MCMC fitting
def model(st, orbits_ZP=0,Amp=Amp_true, phi0=phi0_true, freq=f_true, gw_beta=gw_beta_true, gw_lambda=gw_lambda_true, t0=orbits_t0+10):
    
    # Create random filename to allow for multiprocessing
    gwfn = 'gws_spam/gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'
    
    # Amp, phi0 = theta[0:Ngalbins], theta[Ngalbins:2*Ngalbins]
    
    # Generate GW signals
    for a, f, p, beta, lamb in zip(Amp, freq, phi0, gw_beta, gw_lambda):
        GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=t0, gw_beta=beta-orbits_ZP, gw_lambda=lamb, dt=1/fs, size=size+300)
        # Amplitude is a/f since we have to convert from_gws to from_instr by differentiating and don't want extra factors of 2pi*f
        GalBin.write(gwfn)
    
    # rawdata = Data.from_gws( 'gw_tmp.h5', orbit_path)
    rawdata = Data.from_gws(gwfn, orbit_path,interpolate=True)
    mA = Afunc(rawdata.measurements)[discard:]
    mE = Efunc(rawdata.measurements)[discard:]
    # If we only fit signal parameters, we don't include T since it has by definition no signal.
    #T = Tfunc(rawdata.measurements)[discard:]
    
    mt = GalBin.t[discard:]
    
    os.remove(gwfn)
    
    # Generate correct amplitude to be compatible with sample data
    nmt = np.copy(mt)[:-1]
    nmA = dphi_to_dnu(fs,mA)
    nmE = dphi_to_dnu(fs,mE)
    
    # Make sure that the model generates data at the correct time
    time_indices = np.where(np.in1d(nmt, st))[0]
    nnmt, nnmA, nnmE = nmt[time_indices], nmA[time_indices], nmE[time_indices]

    return np.array([nnmt,nnmA,nnmE])
    # return np.array([t,A,E,T])

# Testing if model works and plotting 
script_time0 = time.time()

mdata = model(fsdata[0])

script_time1 = time.time() - script_time0
print ("Time to run model once = {:.2f} s / {:.2f} min".format(script_time1,script_time1/60))


if gen_plots:
    fig, axs = plt.subplots(4, figsize=(16,8))#, sharex=True, gridspec_kw={'hspace':0})
    axs[0].set_title("Sample and model data")
    for i,j in zip(range(4),[0,0,1,1]):
        axs[i].plot(fsdata[0]/day,fsdata[j+1],label='Channel '+rec[j])
        axs[i].plot(mdata[0]/day,mdata[j+1],label='Channel '+rec[j])
        axs[i].legend(loc=1)
        axs[i].set_ylabel('Amplitude')
        axs[i].set_xlabel('Time (d)')
        if i%2 == 0:
            axs[i].set_xlim(0,duration//day)
        else:
            axs[i].set_xlim(.25,.5)
    plt.savefig("plots/Sample+Model_AETdata.jpg")


sigma = parabula_err(0,.01)
