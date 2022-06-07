# By Bart Ouwehand 12-04-2022
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn

from tqdm import tqdm
# import seaborn as sns
# plt.style.use('seaborn')


# duration = np.array([1,5,10,30,60])
# alpha = np.array([1,10,100,1000,10000])
# duration = np.array([1,5,10,30])
fs = 0.05 #hz
day = 86400 #s
# duration = np.array([1,5,10,30])
duration = np.array([30,90,180])
size = duration*fs*day


# alpha = np.array([1,10,100,1000,10000])
alpha = np.array([1,10,100,1000])

N1, N2 = 10, 31 #number of samples and length of range iteration

calc_sigs = False


if calc_sigs:
    import h5py
    import scipy.signal
    import logging
    import multiprocess

    from lisagwresponse import GalacticBinary
    # from lisainstrument import Instrument

    from pytdi import Data
    # from pytdi import michelson as mich
    from pytdi import ortho

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import BarycentricMeanEcliptic

    # Import own functions
    from my_functions import *


    orbit_path = '../../orbits/esa-orbits.h5'
    gw_path = 'gws.h5'

    discard = 300
    rec = ['A','E','T']

    with h5py.File(orbit_path) as orbits:
        orbits_t0 = orbits.attrs['t0']
        orbit_fs = 1/orbits.attrs['dt']
        orbit_dur = orbits.attrs['tsize']/orbit_fs
        print ("fs = "+str(fs)+" Hz,  orbit_duration = "+str(orbit_dur/day)+" d")

    # Turn on/off binary creation & instrument simulation
    use_verbinaries = True
    tight_range = False

    # Specify specific number of binaries
    Ngalbins = 16

    # Insert binary parameters
    amplitude_amplification = 10

    if use_verbinaries:
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

    # Defining the model used for MCMC fitting
    def model(st,orbits_ZP,Afunc,Efunc,s,Amp=Amp_true, phi0=phi0_true, freq=f_true, gw_beta=gw_beta_true, gw_lambda=gw_lambda_true, t0=orbits_t0+1/fs):

        # Create random filename to allow for multiprocessing
        gwfn = 'gws_spam/gwtmp_'+str(int(1e20*np.random.rand(1)[0]))+'.h5'

        # Amp, phi0 = theta[0:Ngalbins], theta[Ngalbins:2*Ngalbins]

        # Generate GW signals
        for a, f, p, beta, lamb in zip(Amp, freq, phi0, gw_beta, gw_lambda):
            GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=t0, gw_beta=beta-orbits_ZP, gw_lambda=lamb, dt=1/fs, size=s+300)
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


    # Define the likelyhood functions
    def lnL(theta, t, y1, y2,Afunc,Efunc,s):
        """
        The log likelihood of our simplistic Linear Regression. 
        """
        # Amp, f, phi0 = theta
        # beta_ZP, Amp = theta[0], theta[1:]
        # Amp_lnL, phi0_lnL = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]
        beta_ZP_lnL = np.copy(theta)

        # newt, y1_model, y2_model = model(t, Amp_lnL,phi0_lnL)
        newt, y1_model, y2_model = model(t, beta_ZP_lnL,Afunc,Efunc,s)

        return 0.5*(np.sum((y1-y1_model)**2)) + 0.5*(np.sum((y2-y2_model)**2))

    def lnprior(theta):
        """
        Define a gaussian prior that preferences values near the observed values of the Galactic binaries     
        """
        # Amp, phi0 = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]

        # beta_ZP, Amp = theta[0], theta[1:]
        # Amp_lnprior, phi0_lnprior = theta[:Ngalbins], theta[Ngalbins:2*Ngalbins]
        beta_ZP_lnprior = np.copy(theta)

        # if (int(np.sum((1e-26 < Amp)*(Amp<1e-20))) == Ngalbins):# and -np.pi <= beta_ZP <= np.pi:
        # if (int(np.sum((1e-26 < Amp_lnprior)*(Amp_lnprior<1e-20))) == Ngalbins) and int(np.sum((-np.pi <= phi0_lnprior)* (phi0_lnprior<= np.pi))) == Ngalbins:
        if -np.pi <= beta_ZP_lnprior <= np.pi:
            return 0
        return np.inf

        # if 1e-18 < Amp < 1e-14:
        #     return gauss_prior(f, obs_q[1]*obs_qe[1],obs_q[1])
        # return -np.inf
        # if int(np.sum((-np.pi <= phi0) * (phi0 <= np.pi))) == Ngalbins:
        #     return np.sum(gauss_prior(Amp,Amp_prior[1],Amp_prior[0])) #gauss_prior(Amp, obs_q[0]*obs_qe[0], obs_q[0]) + gauss_prior(f, obs_q[1]*obs_qe[1],obs_q[1])
        # return -np.inf

    def lnprob(theta, t, y1, y2,Afunc,Efunc,s):
        """
        The likelihood to include in the MCMC.
        """
        # global iters
        # iters +=1

        lp = lnprior(theta)
        if not np.isfinite(lp):
            # print (iters,'infty')
            return np.inf
        lnlikely = lp + lnL(theta,t,y1, y2,Afunc,Efunc,s)
        # print (iters,lnlikely)
        return lnlikely

    # Define the parabula functions

    def sig(x0,xpm,N,f0,fpm):
        return xpm / np.sqrt(2*np.log((N-f0) / (N-fpm)))

    def parabula_err(x0, xpm, fsdata,L_on_range,f0,Afunc,Efunc,s):
        """Finds error of likelyhood parabula. Input must be the optimal value and the offset from this for which to calculate the parabula. The output is sigma"""
        med = np.median(L_on_range)
        # f0 = lnprob([x0],fsdata[0],fsdata[1], fsdata[2])
        fp, fm = lnprob([x0+xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s),lnprob([x0-xpm],fsdata[0],fsdata[1], fsdata[2],Afunc,Efunc,s)
        # print (med, f0, fp, fm)
        return np.nanmean([sig(x0,xpm,med,f0,fp), sig(x0,xpm,med,f0,fm)])

    def parabula(x0, L_on_range,f0,sig):
        N = np.median(L_on_range)
        # A = lnprob([x0],fsdata[0],fsdata[1], fsdata[2]) - N
        length = 2000
        xrange = np.linspace(-np.pi,np.pi,length)
        return xrange, N - (N-f0)*np.exp(-((xrange-x0)**2)/(2*(sig**2)))
        # return xrange, N + (f0-N)*np.exp(((xrange-x0)**2)/(sig**2))

        
if calc_sigs:
    # Making plots for 
    fp = 'measurements/tm_asds/'
    zp_range = np.pi*np.linspace(-1,1,N2)
    # zp_range_tight = 0.25*np.linspace(-1,1,N2)

    sigmas = np.zeros((len(duration),len(alpha),N1))

    for d,dr in enumerate(duration):
        print ("Calculations for duraiton {} d".format(dr))
        if calc_sigs:
            Afunc, Efunc = BuildModelTDI(orbit_path,fs,size[d],Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True)
        
        sigmas_1d = np.zeros((len(alpha),N1))
        # for a,alph in enumerate(tqdm(alpha)):
        for a,alph in enumerate(alpha):
            # Start by reading in the likelyhood range
            file1 = fp+str(dr)+'d/L_range_a'+str(alph)+'.txt'
            rawdata1 = ascii.read(file1)
            L_range = np.zeros((N1,N2))
            for i in range(N1):
                L_range[i] = np.array(rawdata1['col'+str(i+1)])[1:]
                
            # if dr != 60:
            #     file2 = fp+str(dr)+'d/L_range_a'+str(alph)+'tight.txt'
            #     rawdata2 = ascii.read(file2)
            #     L_range_tight = np.zeros((N1,N2))
            #     for i in range(N1):
            #         L_range_tight[i] = np.array(rawdata2['col'+str(i+1)])[1:]
                    
            # for i in range(N1):
            #     plt.figure(figsize=(8,6))
            #     plt.plot(zp_range, L_range[i], marker='.', linestyle='--', linewidth=1, c='b')
            #     plt.plot(zp_range_tight, L_range_tight[i], marker='.', linestyle='-', linewidth=1, c='r')
            #     plt.title("Likelyhood range for {} d and {} alpha value".format(dr,alph))
            #     plt.xlabel("Orbit-ZP [rad]")
            #     plt.ylabel("-ln(L)")
            #     plt.yscale('log')
            #     plt.savefig('plots/tight/'+str(dr)+'d/'+str(alph)+'/'+str(i)+'.jpg')
                # plt.close()
            
            sigmas_1a = np.zeros(N1)
            fp2 = 'plots/'+str(dr)+'d/'+str(int(alph))+'/'

            for j in range(N1):
                # inputf = '../../data1/measurements/tm_asds/'+str(int(dr))+'d/'+str(int(alph))+"/fs"+str(j)
                inputf = 'measurements/tm_asds/'+str(int(dr))+'d/'+str(int(alph))+"/fs"+str(j)

                rawdata = ascii.read(inputf+'.txt')
                fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

                sigma = parabula_err(0,0.01, fsdata, L_range[j], L_range[j,N2//2],Afunc,Efunc,size[d])
                x,y = parabula(0,L_range[j], L_range[j,N2//2],sigma)
                # print ("Time to calculate likelyhood range = {:.2f} s / {:.3f} hrs".format(time.time()-L_range_time0,(time.time()-L_range_time0)/3600))

                sigmas_1a[j] = sigma
                # print ("For alpha={}, sigma={:.2f}".format(alph,sigma))
                
                plt.figure(figsize=(8,6))
                plt.plot(x,y,c='black',alpha=.8)
                plt.plot(zp_range,L_range[j],marker='.',ls='--',c='r',linewidth=1)
                # if dr != 60:
                #     plt.plot(zp_range_tight,L_range_tight[j],marker='o',ls='--',c='indianred')
                plt.xlabel("zp [rad]")
                plt.ylabel("-ln(L)")
                plt.title("Likelyhood range over ZP range for alpha {} and iteration {}".format(alph,j))
                # plt.ylim(np.min(L_range),np.max(L_range))
                plt.savefig(fp2+"L_range_over_ZP_"+str(j)+".jpg")
                plt.close()

            print ("sigmas for duration",dr,"d, alpha",alph,":",sigmas_1a)
            sigmas_1d[a] = sigmas_1a
        sigmas[d] = sigmas_1d
        filename = fp+str(dr)+'d/sigmas.txt'
        
        print (sigmas_1d.shape,alpha.shape)
        filecontent = Table(sigmas_1d.T,names=alpha)
        ascii.write(filecontent, filename, overwrite=True)


sigmas = np.zeros((len(duration),len(alpha),N1))
for d,dr in enumerate(duration):
    filename= 'measurements/tm_asds/'+str(dr)+'d/sigmas.txt'
    rawdata = ascii.read(filename)
    sigmas_1d = np.zeros((len(alpha),N1))
    for a,alph in enumerate(alpha):
        sigmas_1d[a] = np.array(rawdata['col'+str(a+1)][1:])
    sigmas[d] = sigmas_1d

# Make the plots of the likelyhood range
def parabula(x0, L_on_range,f0,sig):
        N = np.median(L_on_range)
        # A = lnprob([x0],fsdata[0],fsdata[1], fsdata[2]) - N
        length = 2000
        xrange = np.linspace(-np.pi,np.pi,length)
        return xrange, N - (N-f0)*np.exp(-((xrange-x0)**2)/(2*(sig**2)))
        # return xrange, N + (f0-N)*np.exp(((xrange-x0)**2)/(sig**2))

zp_range = np.pi*np.linspace(-1,1,N2)
for d,dr in enumerate(duration):
    fp = 'plots/'+str(dr)+'d/'
    print ("Making plots for {} d".format(dr))
    for a,alph in enumerate(tqdm(alpha)):
        # Start by reading in the likelyhood range
        file1 = 'measurements/tm_asds/'+str(dr)+'d/L_range_a'+str(alph)+'.txt'
        rawdata1 = ascii.read(file1)
        fpp = fp+str(alph)+'/'
        for i in range(N1):
            L_range = np.array(rawdata1['col'+str(i+1)])[1:]
            x,y = parabula(0,L_range, L_range[N2//2],sigmas[d,a,i])
            plt.figure(figsize=(8,6))
            plt.plot(x,y,c='black',alpha=.8)
            plt.plot(zp_range,L_range,marker='.',ls='--',c='r',linewidth=1)
            plt.xlabel("zp [rad]")
            plt.ylabel("-ln(L)")
            plt.title("Likelyhood range over ZP range for alpha {} and iteration {}".format(alph,i))
            # plt.ylim(np.min(L_range),np.max(L_range))
            plt.savefig(fp+str(alph)+"/L_range_over_ZP_"+str(i)+".jpg")
            plt.close()

skip = 0
plt.figure(figsize=(8,6))
for d,dr in enumerate(duration[skip:]):
    plt.errorbar(alpha,np.nanmean(sigmas[skip+d],axis=1),yerr = np.nanstd(sigmas[skip+d],axis=1),fmt='--o',label=str(dr)+'d',linewidth=1,capsize=5,capthick=1)
    print ("For",str(dr),"d, number of non-nan points = ",np.sum(~np.isnan(sigmas[skip+d]),axis=1))
plt.ylabel("$\sigma_{\delta}$")
plt.xscale('log')
plt.legend()
plt.xlabel("TM_acceleration noise amplification")
plt.title("Error in determining orbits ZP ($\delta$) for amplified tm acceleration noise")
plt.savefig('plots/masterplot.jpg')
plt.close()