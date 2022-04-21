# By Bart Ouwehand 12-04-2022
import numpy as np
import matplotlib.pyplot as plt

import h5py
import logging
import time
import os

from lisagwresponse import GalacticBinary
from lisainstrument import Instrument

from pytdi import Data
from pytdi import michelson as mich
from pytdi import ortho

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn

def GenerateGalbins(orbit_path,gw_path, fs, size, Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true, t0):
    """Generates a new file gw_path with N galactic binaries. Input is orbit_path(str): path to orbits file, gw_path(str): filepath+extension, fs(float): sampling frequency, size(int): size of samples, Amp_true(N-array): gw amplitudes, f_true(N-array): gw frequencies, phi0_true(N-array): gw initial phases, gw_beta_true(N-array): gw galactic longitude coords, gw_lambda_true(N-array): gw galactic lattitude, t0(N-array): start of signal"""
    print ("Generating gravitational waves")
    if os.path.exists(gw_path):
        os.remove(gw_path)
    for a,f,p,beta,lamb in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true):
        source = GalacticBinary(A=a, f=f, phi0=p, orbits=orbit_path ,t0=t0, gw_beta=beta, gw_lambda=lamb, dt=1/fs, size=size+300)
        source.write(gw_path)

def GenerateInstrumentAET(orbit_path, gw_path, fs, size, sample_outputf, discard, genTDI=True, sAfunc = False, sEfunc = False, sTfunc = False, tm_alpha=1):
    # Setup logger (sometimes useful to follow what's happening)
    # logging.basicConfig()
    # logging.getLogger('lisainstrument').setLevel(logging.INFO)
    print ("Starting simulation")
    
    tm_asds = { k: tm_alpha*2.4e-15 for k in Instrument.MOSAS}
    # tm_asds['31'] = tm_alpha*2.4e-15
    
    t00 = time.time()
    sample_instru = Instrument(
        size=size, # in samples
        dt=1/fs,
        aafilter=('kaiser', 240, 0.275*fs, 0.725*fs),
        orbits=orbit_path, # realistic orbits (make sure it's consistent with glitches and GWs!)
        gws=gw_path,
        testmass_asds=tm_asds
    )
    # sample_instru.disable_all_noises()
    sample_instru.simulate()
    
    
    # Write out data to sample file, NOTE: Remember to remove the old sample file.
    if os.path.exists(sample_outputf+'.h5'):
        os.remove(sample_outputf+'.h5')
    sample_instru.write(sample_outputf+'.h5')
    
    
    # Read data from LISA Instrument
    rawdata = Data.from_instrument(sample_outputf+'.h5')
    
    if genTDI:
        print ("Time to run simulation = {:.2f} s / {:.3f} hrs".format((time.time()-t00),(time.time()-t00)/3600))
        t0 = time.time()
        sAfunc = ortho.A2.build(**rawdata.args)
        A = sAfunc(rawdata.measurements)[discard:]
        t1 = time.time()
        print ("Time to build and run A2 = {:.2f} s / {:.3f} hrs".format((t1-t0),(t1-t0)/3600))
        sEfunc = ortho.E2.build(**rawdata.args)
        E = sEfunc(rawdata.measurements)[discard:]
        t2 = time.time()
        print ("Time to build and run E2 = {:.2f} s / {:.3f} hrs".format((t2-t1),(t2-t1)/3600))
        sTfunc = ortho.T2.build(**rawdata.args)
        T = sTfunc(rawdata.measurements)[discard:]
        t3 = time.time()
        print ("Time to build and run T2 = {:.2f} s / {:.3f} hrs".format((t3-t2),(t3-t2)/3600))
    else:
        t0 = time.time()
        A = sAfunc(rawdata.measurements)[discard:]
        t1 = time.time()
        # print ("Time to run A2 = {:.2f} s / {:.3f} hrs".format((t1-t0),(t1-t0)/3600))
        E = sEfunc(rawdata.measurements)[discard:]
        t2 = time.time()
        # print ("Time to run E2 = {:.2f} s / {:.3f} hrs".format((t2-t1),(t2-t1)/3600))
        T = sTfunc(rawdata.measurements)[discard:]
        t3 = time.time()
        # print ("Time to run T2 = {:.2f} s / {:.3f} hrs".format((t3-t2),(t3-t2)/3600))
    
    t = sample_instru.t[discard:]
    # t = (np.arange(0,len(A)+discard)/fs)[discard:]

    sdata = np.array([t,A,E,T])

    # Extract A, E, T data to speed up re-running code.
    filepath = sample_outputf+'.txt'
    filecontent = Table(sdata.T, names=['t','A','E','T'])
    ascii.write(filecontent, filepath, overwrite=True)

    print ("Total time for sample = {:.2f} s / {:.2f} hrs".format(time.time()-t00,(time.time()-t00)/3600))
    
    if genTDI:
        return sAfunc, sEfunc, sTfunc

def dphi_to_dnu(fs,data):
    laser_freq = 2.816E14 #Hz, gotten from lisainstrument code
    dt = 1/fs
    # dt = np.mean((time[1:]-time[:-1]))
    return np.diff(data) * ((laser_freq) / (2*np.pi*dt))

def BuildModelTDI(orbit_path,fs,size,Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True,interpolate=True):
    """Build the TDI channels for the model data"""
    # Generate random binary to be able to build the TDI chanels

    if os.path.exists('gws_tmp.h5'):
        os.remove('gws_tmp.h5')
    
    # Keep track of time
    time_elapsed = []
    
    with h5py.File(orbit_path) as orbits:
        orbits_t0 = orbits.attrs['t0']
    if detailed:
        time_elapsed.append(time.time())
        for a, f, p, beta, lamb in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true):
            GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=orbits_t0+10, gw_beta=beta, gw_lambda=lamb, dt=1/fs, size=size+300)
            GalBin.write('gw_tmp.h5')
    
        time_elapsed.append(time.time()) 
        rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=interpolate)#, skipped=-int(size))
        os.remove('gw_tmp.h5')
    
    else:
        GalBin = GalacticBinary(A=1e-20, f=1e-3, phi0=0, orbits=orbit_path, t0=orbits_t0+10, gw_beta=0, gw_lambda=0, dt=1/fs, size=size+300)
        GalBin.write('gw_tmp.h5')
        rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=interpolate)#, skipped=-int(size))
        os.remove('gw_tmp.h5')
    
    time_elapsed.append(time.time())

    print ("Start model TDI building")

    Afunc = ortho.A2.build(**rawdata.args)
    Efunc = ortho.E2.build(**rawdata.args)
    time_elapsed.append(time.time())
    A = Afunc(rawdata.measurements)[discard:]
    E = Efunc(rawdata.measurements)[discard:]
    time_elapsed.append(time.time())

    tmp = np.array(time_elapsed)[1:] - np.array(time_elapsed[:-1])
    if detailed:
        print ("For model data: gw_gen = {:.2f} s, signal_gen = {:.2f} s, TDIbuild = {:.2f} s, TDIcalc = {:.2f} s, Total = {:.2f} s".format(*tmp,time_elapsed[-1]-time_elapsed[0]))
    else: 
        print ("For model data: TDIbuild = {:.2f} s, TDIcalc = {:.2f} s, Total = {:.2f} s".format(*tmp,time_elapsed[-1]-time_elapsed[0]))
    
    return Afunc, Efunc



