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
from lisaconstants import c


from pytdi import Data
from pytdi import michelson as mich
from pytdi import ortho
from pytdi.dsp import timeshift

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import BarycentricMeanEcliptic

from tqdm import tqdm

# Import own functions
from my_functions import *


orbit_path = '../../orbits/esa-orbits.h5'
gw_path = 'gws.h5'

# Setup simluation parameters
fs = 0.1    # Hz
# fs = 0.05    # Hz
day = 86400 # s
duration = day*1 # X days
size = duration*fs
discard = 300
rec = ['A','E']

with h5py.File(orbit_path) as orbits:
    orbits_t0 = orbits.attrs['t0']
    orbit_fs = 1/orbits.attrs['dt']
    orbit_dur = orbits.attrs['tsize']/orbit_fs
    print ("fs = "+str(fs)+" Hz,  orbit_duration = "+str(orbit_dur/day)+" d")

# Turn on/off binary creation & instrument simulation
use_verbinaries = True


# Specify specific number of binaries
Ngalbins = 16

# Insert binary parameters
amplitude_amplification = 1

if use_verbinaries:
    # Define name of simulation uitput file
    sample_outputf = 'measurements/sampdat_'+str(duration//day)+'d'+'_verbins' #extention of .h5 or .txt added later
    
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

from lisainstrument.containers import ForEachMOSA, ForEachSC

class testInstrument(Instrument):
    def __init__(self, **kw):
        super().__init__( laser_asds=0, 
                          modulation_asds=0,
                          aafilter=('kaiser', 240, 0.275*fs, 0.725*fs),
                          oms_asds =(15e-12, 1.25e-11, 1.42e-12, 3.38e-12, 3.32e-12, 7.9e-12),
                          testmass_asds = 3e-15,
                          lock='six', orbits='static', **kw)
        
        self.disable_clock_noises()
        self.disable_all_noises()

        
    def init_orbits(self, orbits, orbit_dataset, tau=2.5e9/c):
       
        if orbits == 'static':
            # logger.info(f"Using default set of static proper pseudo-ranges L/c={tau}")
            self.orbit_file = None
            self.pprs = ForEachMOSA({
                # Default PPRs based on first samples of Keplerian orbits (v1.0)
                '12': tau, '23': tau, '31': tau,
                '13': tau, '32': tau, '21': tau,
            })
            self.d_pprs = ForEachMOSA(0)
            self.tps_proper_time_deviations = ForEachSC(0)
            self.orbit_dataset = None
        else:
            super().init_orbit(orbits, orbit_dataset )

def model2(Afunc,Efunc,s, Amp=1, phi0=phi0_true[0], freq=f_true[0], gw_beta=gw_beta_true[0], gw_lambda=gw_lambda_true[0], t0=orbits_t0+1/fs):
    
    # Create random filename to allow for multiprocessing
    gwfn = 'gws_spam/gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'
    
    GalBin = GalacticBinary(A=Amp, f=freq, phi0=phi0, orbits=orbit_path, t0=t0, gw_beta=gw_beta, gw_lambda=gw_lambda, dt=1/fs, size=s+300)
    GalBin.write(gwfn)
    
    Instrument = testInstrument(dt=1/fs, size=s+300,gws=gwfn,t0=orbits_t0)
    
    Instrument.simulate()
    # Instrument.write(mode='w')
    
    # rawdata = Data.from_gws( 'gw_tmp.h5', orbit_path)
    rawdata = Data.from_instrument(Instrument)#,interpolate=True)
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
    # time_indices = np.where(np.in1d(nmt, st))[0]
    # nnmt, nnmA, nnmE = nmt[time_indices], nmA[time_indices], nmE[time_indices]
    
    return np.array([nmt,nmA,nmE])
    # return np.array([t,A,E,T])

# Generate A and E TDI chanels for perfect model
if os.path.exists('gwtmp.h5'):
    os.remove('gwtmp.h5')
source = GalacticBinary(A=1, f=f_true[0], phi0=phi0_true[0], orbits=orbit_path,
                        t0=orbits_t0 + 10, gw_beta=gw_beta_true[0], gw_lambda=gw_lambda_true[0],
                        dt=1/fs, size=size+300)
source.write('gwtmp.h5')
tmp_instru = testInstrument(dt=1/fs, size=size+300,gws='gwtmp.h5')
tmp_instru.simulate()
rawdata = Data.from_instrument(tmp_instru)
Afunc = ortho.A2.build(**rawdata.args)
Efunc = ortho.E2.build(**rawdata.args)
os.remove('gwtmp.h5')

# Define psd function
def psd_func(data):
    return np.array(scipy.signal.welch(data,fs=fs,window='nuttall',nperseg=len(data),detrend=False))

# Define FFT functions
def FFT(A):
    from numpy.fft import fft, fftshift, ifft, ifftshift
    fourierA = fftshift(fft(A))
    return fourierA

def IFFT(fourierA):
    from numpy.fft import fft, fftshift, ifft, ifftshift
    A = ifft(ifftshift(fourierA))
    return A

def make_psd(data):
    tmp = []
    for i in range(1,3):
        psdtmp = psd_func(data[i])
        tmp.append(psdtmp[1])
    psd = np.array([psdtmp[0],*tmp])
    return psd

def make_fft(data):
    tmp = []
    fft_freq = np.fft.fftshift(np.fft.fftfreq(len(data[1]), d=1/fs))
    fft_freq = fft_freq[len(fft_freq)//2:]
    for i in range(1,3):
        ffttmp = FFT(data[i])
        tmp.append(ffttmp[len(ffttmp)//2:]*2)
    fft = np.array([fft_freq, *tmp])
    return fft[:,1:]

def psd_tdi(f, channel='X', L=2.5e9, s_oms = 15e-12, s_acc = 3e-15):
    def C(x):
        return 16*np.sin(x)**2*np.sin(2*x)**2
    
    def C1(x):
        return 16*np.sin(3/2*x)**2
    
    def C2(x):
        return 16*np.sin(x/2)**2
    
    w = 2*np.pi*f
    x=w*L/c
    lam = 1.064e-6 
    
    # keep in mind that we are looking at frequency fluctuations  
    # df = d \dot phi = \dot (dx/lambda) = w*dx/lambda
    
    dx_to_dnu = (w/lam)
    
    # The Otpical Metrology Noise (OMS) has a frequency shape factor that 
    # allows a ramp-up of the noise for low frequency. That is not necessarily a 
    # prediction of the behaviour, but more a relaxation factor.
    # It relaxes at 2mHz which is about the point where LISA's sensitivity is dominated
    # by acceleration noise anyway. So there is no need to be ultra-strict with the 
    # requirements.
    
    S_oms = s_oms**2 * (1. + (2.e-3/f)**4)
    S_oms_nu = S_oms*dx_to_dnu**2
    
    # The acceleration noise (ACC) has two shaping factors. One at 
    # 8mHz where it is allowed to rise up, as at that frequency we are dominated by
    # OMS noise anyway, so no need to be strict at high frequencies
    #
    # The other factor *reddends* the noise below 0.4mHz. This is more relevant as
    # it assumes that at low frequencies we find some effects that make the acceleration noise worse.
    # Additional shaping factors at lower frequencies are possible, but are not covered in the 
    # Science Requirements.
    
    S_acc= s_acc**2 * (1.0 +(0.4e-3/f)**2)*(1.0+(f/8e-3)**4)
    S_acc_nu = S_acc*dx_to_dnu**2
    
    # we only implement the H_1 and H2 terms here (see Antoines note)
    if channel in ['X','Y','Z']:
        return 4*C(x)*( S_oms_nu+ (3+np.cos(2*x))*S_acc_nu/(w**4))
    elif  channel in ['A','E']:
        return 2*C(x) * ( ( 2+np.cos(x))*S_oms_nu + 2*( ( 3+2*np.cos(x) + np.cos(2*x) ) )*S_acc_nu/(w**4) )
    elif  channel in ['T']:
        return 4*C(x)*( (1-np.cos(x))*S_oms_nu +  8*np.sin(0.5*x)**4*S_acc_nu/(w**4) )
    elif channel in ['α', 'β', 'γ']:
        # guesswork
        return 2*C1(x) * (1/np.sqrt(2)*S_oms_nu+ np.sin(x)**2*(3+np.cos(2*x))*S_acc_nu/(w**4))
    elif channel in ['ζ_1', 'ζ_2', 'ζ_3']:
        # guesswork
        return 2*C2(x) * (1/np.sqrt(2)*S_oms_nu+ np.sin(x)**2*(3+np.cos(2*x))*S_acc_nu/(w**4))
    else: 
        assert False, "Only channel X,Y,Z, or  A, E, T are currently implemented"


# Calculate the fft of the individual binary signals
fft_sigs = []
print ("Calculate the ffts for all binaries")
for i in tqdm(range(Ngalbins)):
    fft_sigs.append(make_fft(model2(Afunc,Efunc,size,phi0=phi0_true[i],
                                freq=f_true[i], gw_beta=gw_beta_true[i],
                                gw_lambda=gw_lambda_true[i])))
alpha_N = 100
reduction_factor = 1e65
alpha = np.logspace(-1,2,alpha_N)
det_covns = {}
for ch in rec:
    det_covns[ch] = np.zeros(((Ngalbins-1),alpha_N))

for n in range(1,Ngalbins):
    det_cov = {}
    for chan in rec:
        det_cov[chan] = np.zeros(alpha_N)

    fishermatricesA = np.zeros((alpha_N,n,n))
    fishermatricesE = np.zeros((alpha_N,n,n))
    print ("Calculate the matrices for {} galbins".format(n))
    for a,alph in enumerate(tqdm(alpha)):
        fisher = {}
        for chan in rec:
            fisher[chan] = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                # fft_i = make_fft(model2(Afunc,Efunc,size,phi0=phi0_true[i],
                #                         freq=f_true[i], gw_beta=gw_beta_true[i],
                #                         gw_lambda=gw_lambda_true[i]))
                # fft_j = make_fft(model2(Afunc,Efunc,size,phi0=phi0_true[j],
                #                         freq=f_true[j], gw_beta=gw_beta_true[j],
                #                         gw_lambda=gw_lambda_true[j]))
                fft_i = fft_sigs[i]
                fft_j =  fft_sigs[j]
                Sn = psd_tdi(np.abs(fft_i[0]), channel='A',s_acc = alph*3e-15) # take frequencies of fft, A & E are identical

                df = fft_i[0][1]-fft_i[0][0]

                # Remove stuff above 10mHz since this is outside of the range of the signals
                highfreqmask = fft_i[0] < 10e-3 #Hz

                for k,chan in enumerate(rec):
                    numerator = (fft_i[k+1]*np.conj(fft_j[k+1]) + np.conj(fft_i[k+1])*fft_j[k+1])
                    denominator = Sn
                    # print ("n,d",numerator, denominator)     
                    element = np.real(np.nansum((numerator/denominator)[highfreqmask]*df))
                    fisher[chan][i,j] = element/reduction_factor
        for chan in rec:
            det_cov[chan][a] = 1/np.linalg.det(fisher[chan])
        fishermatricesA[a] = fisher['A']
        fishermatricesE[a] = fisher['E']

    # Calculate scaled determinant of covariance matrix
    index_closest_to1 = np.where((alpha-1)**2==np.min((alpha-1)**2))[0][0]
    for ch in rec:
        det_covns[ch][n-1] = det_cov[ch]/(det_cov[ch][index_closest_to1])

for ch in rec:
    plt.figure(figsize=(8,6))
    plt.axvline(1,c='black',linestyle='--',alpha=.5)
    plt.axhline(1,c='black',linestyle='--',alpha=.5)
    for n in range(1,Ngalbins):
        plt.loglog(alpha,det_covns[ch][n-1]**(1/n),label=n)
        # plt.loglog(alpha,det_cov[chan],label=chan)
    plt.xlabel("tm acceleration noise amplification")
    plt.ylabel("Det of covariance matrix")
    plt.title(ch)
    plt.legend()
    plt.show()