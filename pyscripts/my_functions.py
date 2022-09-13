# By Bart Ouwehand 08-06-2022
from config import *

def GenerateGalbins(orbit_path,gw_path, fs, size, Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true, iota_true, t0):
    """Generates a new file gw_path with N galactic binaries. Input is orbit_path(str): path to orbits file, gw_path(str): filepath+extension, fs(float): sampling frequency, size(int): size of samples, Amp_true(N-array): gw amplitudes, f_true(N-array): gw frequencies, phi0_true(N-array): gw initial phases, gw_beta_true(N-array): gw galactic longitude coords, gw_lambda_true(N-array): gw galactic lattitude, t0(N-array): start of signal"""
    # print ("Generating gravitational waves")
    if os.path.exists(gw_path):
        os.remove(gw_path)
    for gba, gbf, gbp, gbbeta, gblamb, iota in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true, iota_true):
        source = GalacticBinary(A=gba, f=gbf, phi0=gbp, orbits=orbit_path ,t0=t0, gw_beta=gbbeta, gw_lambda=gblamb, iota=iota, dt=1/fs, size=size+300)
        # source = GalacticBinary(A=gba, f=gbf, phi0=gbp, orbits=orbit_path ,t0=t0, gw_beta=gbbeta, gw_lambda=gblamb, dt=1/fs, size=size+300)
        source.write(gw_path)

# def GenerateInstrumentAET(orbit_path, gw_path, fs, size, sample_outputf, discard, genTDI=True, sAfunc = False, sEfunc = False, sTfunc = False, tm_alpha=1):
def GenerateInstrumentAET(orbit_path, gw_path, fs, size, discard, genTDI=True, sAfunc = False, sEfunc = False, tm_alpha=1, OMS_alpha=1, noise=True, printing=True,turnoff = None):
    # Setup logger (sometimes useful to follow what's happening)
    # logging.basicConfig()
    # logging.getLogger('lisainstrument').setLevel(logging.INFO)
    if printing:
        print ("Starting simulation")
    
    tm_asds = { k: np.sqrt(tm_alpha)*2.4e-15 for k in Instrument.MOSAS}
    # tm_asds['31'] = tm_alpha*2.4e-15
    
    t00 = time.time()
    sample_instru = Instrument(
        size=size, # in samples
        dt=1/fs,
        aafilter=('kaiser', 240, 0.275*fs, 0.725*fs),
        orbits=orbit_path, # realistic orbits (make sure it's consistent with glitches and GWs!)
        gws=gw_path,
        testmass_asds=tm_asds,
        oms_asds = np.array([6.35e-12, 1.25e-11, 1.42e-12, 3.38e-12, 3.32e-12, 7.9e-12])*np.sqrt(OMS_alpha)
    )
    if not noise:
        # noises = ['laser', 'modulation', 'clock', 'pathlength', 'ranging', 'jitters']
        # sample_instru.disable_all_noises(but='laser')
        # laser, clock and pathlength noise dominate but only pathlength is influenced by oms and tm_asds noise
        sample_instru.disable_all_noises()
        
    if noise and turnoff is not None:
        if 'laser' in turnoff:
            sample_instru.laser_asds = ForEachMOSA(0)
        if 'modulation' in turnoff:
            sample_instru.modulation_asds = ForEachMOSA(0)
        if 'clock' in turnoff:
            sample_instru.disable_clock_noises()
        if 'pathlength' in turnoff:
            sample_instru.disable_pathlength_noises()
        if 'ranging' in turnoff:
            sample_instru.disable_ranging_noises()
        if 'jitters' in turnoff:
            sample_instru.disable_jitters()
    
    sample_instru.simulate()
    
    
    # Read data from LISA Instrument
    rawdata = Data.from_instrument(sample_instru)
    
    if genTDI:
        print ("Time to run simulation = {:.2f} s / {:.3f} hrs".format((time.time()-t00),(time.time()-t00)/3600))
        t0 = time.time()
        sAfunc = ortho.A2.build(**rawdata.args)
        A = sAfunc(rawdata.measurements)[300:]
        t1 = time.time()
        print ("Time to build and run A2 = {:.2f} s / {:.3f} hrs".format((t1-t0),(t1-t0)/3600))
        sEfunc = ortho.E2.build(**rawdata.args)
        E = sEfunc(rawdata.measurements)[300:]
        t2 = time.time()
        print ("Time to build and run E2 = {:.2f} s / {:.3f} hrs".format((t2-t1),(t2-t1)/3600))
    else:
        t0 = time.time()
        A = sAfunc(rawdata.measurements)[300:]
        t1 = time.time()
        E = sEfunc(rawdata.measurements)[300:]
        t2 = time.time()
    
    t = (sample_instru.t)[300:]
    # t = (np.arange(0,len(A)+discard)/fs)[discard:]

    sdata = np.array([t,A,E])
    
    if printing:
        print ("Total time for sample = {:.2f} s / {:.2f} hrs".format(time.time()-t00,(time.time()-t00)/3600))
    
    # print ("RAM before del statement = {}".format(psutil.virtual_memory()[2]))
    # del sample_instru
    # print ("RAM after del statement = {}".format(psutil.virtual_memory()[2]))
    
    if genTDI:
        # return sAfunc, sEfunc, sTfunc
        return sdata, sAfunc, sEfunc
    else:
        return sdata

def dphi_to_dnu(fs,data):
    laser_freq = 2.816E14 #Hz, gotten from lisainstrument code
    dt = 1/fs
    # dt = np.mean((time[1:]-time[:-1]))
    return np.diff(data) * ((laser_freq) / (2*np.pi*dt))

def BuildModelTDI(orbit_path,fs,size,Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,discard,detailed=True,interpolate=True):
    """Build the TDI channels for the model data"""
    # Generate random binary to be able to build the TDI chanels

    if os.path.exists('gw_tmp.h5'):
        os.remove('gw_tmp.h5')
    
    # Keep track of time
    time_elapsed = []
    
    with h5py.File(orbit_path) as orbits:
        orbits_t0 = orbits.attrs['t0']
    if detailed:
        time_elapsed.append(time.time())
        for a, f, p, beta, lamb in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true):
            if a not in Amp_true[1:]:
                GalBin = GalacticBinary(A=a/f, f=f, phi0=p, orbits=orbit_path, t0=orbits_t0+1/fs, gw_beta=beta, gw_lambda=lamb, dt=1/fs, size=size+300)
                GalBin.write('gw_tmp.h5')
    
        time_elapsed.append(time.time()) 
        # rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=interpolate)#, skipped=-int(size))
        rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=False)#, skipped=-int(size))
        os.remove('gw_tmp.h5')
    
    else:
        GalBin = GalacticBinary(A=1e-20, f=1e-3, phi0=0, orbits=orbit_path, t0=orbits_t0+1+1/fs, gw_beta=0, gw_lambda=0, dt=1/fs, size=size+300)
        GalBin.write('gw_tmp.h5')
        # rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=interpolate)#, skipped=-int(size))
        rawdata = Data.from_gws('gw_tmp.h5',orbit_path,interpolate=False)#, skipped=-int(size))
        os.remove('gw_tmp.h5')
    
    time_elapsed.append(time.time())

    print ("Start model TDI building")

    Afunc = ortho.A2.build(**rawdata.args)
    Efunc = ortho.E2.build(**rawdata.args)
    time_elapsed.append(time.time())
    A = Afunc(rawdata.measurements)[300:]
    E = Efunc(rawdata.measurements)[300:]
    time_elapsed.append(time.time())

    tmp = np.array(time_elapsed)[1:] - np.array(time_elapsed[:-1])
    if detailed:
        print ("For model data: gw_gen = {:.2f} s, signal_gen = {:.2f} s, TDIbuild = {:.2f} s, TDIcalc = {:.2f} s, Total = {:.2f} s".format(*tmp,time_elapsed[-1]-time_elapsed[0]))
    else: 
        print ("For model data: TDIbuild = {:.2f} s, TDIcalc = {:.2f} s, Total = {:.2f} s".format(*tmp,time_elapsed[-1]-time_elapsed[0]))
    
    return Afunc, Efunc

# Define function to take PSD of data
def psd_func(data):
    return scipy.signal.welch(data,fs=fs,window='nuttall',nperseg=len(data),detrend=False)

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


class myGalacticBinary(GalacticBinary):
    def __init__(self, A,f,orbits, gw_beta=0, gw_lambda=0,t0=None,**kwargs):
        # make sure that the parent __init__ is called if orbit is a string
        if isinstance(orbits,str):
            super().__init__(A,f,orbits=orbits, gw_beta=gw_beta, gw_lambda=gw_lambda, **kwargs)
        else:
        # Create the auxilliary dictionaries that hold the interpolating functions:
            pos={}
            x={}
            y={}
            z={}
            tt={}

            # build a copy of the orbit times and extend it to the left and the right to make sure 
            # that the interpolation does not run out of range
            t = np.copy(orbits.t)
            if t0 == None:
                t0 = t[0]

            t = np.insert(t,0, t0-orbits.dt)
            t = np.insert(t,-1, t0+orbits.dt)

            # build the coordinate interpolators
            for sc in orbits.SC_INDICES:
                pos[sc]=orbits.compute_spacecraft_position(sc,t)
                x[f'{sc}']=interp1d(t, pos[sc][:,0])
                y[f'{sc}']=interp1d(t,pos[sc][:,1])
                z[f'{sc}']=interp1d(t,pos[sc][:,2])

            # build the light-travel times for the links
            for links in orbits.LINK_INDICES:
                L = orbits.compute_light_travel_time(int(links[0]),int(links[1]),t)
                tt[links] = interp1d(t, L)

            # now call the parent __init_ with the right parameters
            super().__init__(A, f, orbits='', gw_beta=gw_beta, gw_lambda=gw_lambda,
                             x=x,y=y,z=z,tt=tt, t0=t[1], **kwargs)
            
class myInstrument(Instrument):
    
    def __init__(self,  orbits='static',gws=None, 
                 t0='orbits',
                 physics_upsampling=4,
                 size=2592000, dt=1/4,
                 **kwargs):
        
                # the follwoing code needs to be copied, unfortunately
                # as it is not abstracted as a method
                if t0 == 'orbits':
                    self.t0=orbits.t[1]
                else:
                    self.t0 = t0

                
                # Physics sampling
                self.size = int(size)
                self.dt = float(dt)
                self.fs = 1 / self.dt

                self.physics_upsampling = int(physics_upsampling)
                self.physics_size = self.size * self.physics_upsampling
                self.physics_dt = self.dt / self.physics_upsampling
                self.physics_fs = self.fs * self.physics_upsampling
       
                self.physics_t = self.t0 + np.arange(self.physics_size, dtype=np.float64) * self.physics_dt
        
                # call the parent's init() if both orbits and gws are strings, otherwise
                # deal with it here
                if not isinstance(gws, str) and not isinstance(orbits, str):
                    # if only the orbit is a string, call the parent's init_orbit
                    if isinstance(orbits,str):
                        super().init_orbits(orbits, orbit_dataset)
                    else:
                        # so orbits is a memory structure. In this case we have to call the parent's init_orbit 
                        # with the correct orbits
                        self.orbits = { links: orbits.compute_light_travel_time(int(links[0]),int(links[1]),self.physics_t) for links in orbits.LINK_INDICES }

                    # if only gws is a string, call the parents init_gws() and then pass it on tho __init__
                    if isinstance(gws, str):
                        super().init_gws(gws)
                    else:
                        # so gws is *not* a string, so we assume it is an object
                        self.gws = { links: gws.compute_gw_response(links,self.physics_t)[0] for links in orbits.LINK_INDICES }


                    # now we can call the parent's init() with the correctly formatted arguments    
                    super().__init__(orbits=self.orbits, gws=self.gws,
                                     physics_upsampling=physics_upsampling,
                                     size=size, dt=dt,t0=self.t0,
                                     **kwargs)

                else:
                    super.__init__( orbits=orbits, gws=gws,
                                    physics_upsampling=physics_upsampling,
                                    size=size, dt=dt, t0=self.t0,
                                    **kwargs)
        

class myData(pytdi.Data):
    
    # overload the 'from_gws' method
    @classmethod
    def from_gws(self, gws, orbits, skipped=0, orbit_dataset='tps/ppr', **kwargs):
        """
        gws: can be an object or a list of objects. If a filename is given, it behaves like the normal 'Data' class
        orbit: Acccpets an object as well as a file
        """

        # deal with the standard case
        if isinstance(gws, str):
            return super().from_gws(gws, orbits, **kwargs)
        
        # prepare the measurements under the assumption that gws is either an object or an iterable of an object
        
        try:
             _ = iter(gws)
        except TypeError as te:
            # if it isn't an iterable yet, make it so.
            gws=[gws]
            
        #now fill the new gws with the right values
        # assume that the first element in the listable gws 
        # has the values we want. We might want to make that
        # a parameter. Later.
        
        fs = gws[0].fs
        t0 = gws[0].t0
        
        t = gws[0].t
        
        # define and preset the measurements
        measurements = {}
        for mosa in self.MOSAS:
            measurements[f'isc_{mosa}'] = 0
            measurements[f'isc_sb_{mosa}'] =0 
            measurements[f'tm_{mosa}'] = 0
            measurements[f'ref_{mosa}'] = 0
            measurements[f'ref_sb_{mosa}'] = 0

        # run through the MOSAS and loop through the iterable
        # adding up the signals
        
        for mosa in self.MOSAS:
            for gws_s in gws:
                data = gws_s.compute_gw_response( mosa, t)[0]
                measurements[f'isc_{mosa}'] += super().slice(data, skipped)
                measurements[f'isc_sb_{mosa}'] += super().slice(data, skipped)
            
        # Load delays from orbit file
        return self.from_orbits(orbits, fs, t0, orbit_dataset, **measurements)
    
    
    # we also need to patch the from_orbits to allow an object to be passed
    @classmethod
    def from_orbits(self, orbits , fs, t0='orbits', dataset='tps/ppr', **kwargs):
        
        # Check if 'orbits' is a string - if so, pass it on the parent class' function
        if isinstance(orbits,str):
            return super().from_orbits(orbits, fs, t0=t0, dataset=dataset, **kwargs)
        
        # if not, assume orbits is an object and fill the parametrs accordingly
        
        # the next few lines are stolen from the parent's from_orbits
        # Check that we have at least one measurement
        if not kwargs:
            raise ValueError("from_orbits() requires at least one measurement")
        # Check that keywords are valid measurements
        size = 0
        for key, arg in kwargs.items():
            if isinstance(arg, (int, float)):
                size = max(size, 1)
            else:
                size = max(size, len(arg))
            if key not in self.MEASUREMENTS:
                raise TypeError(f"from_orbits() has invalid measurement key '{key}'")
                
        if t0=='orbits':
            t0 = orbits.t0
            
        t = t0 + np.arange(size) / fs
        
        delays={}
        for i, mosa in enumerate(self.MOSAS):
            delays[f'd_{mosa}'] = orbits.compute_light_travel_time(int(mosa[0]), int(mosa[1]),t)
            
        # Create measurements dictionary
        measurements = {key: kwargs.get(key, 0) for key in self.MEASUREMENTS}
        # Create instance
        data = self(measurements, delays, fs)
        data.compute_delay_derivatives()
        return data

##################################################################################################################
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Container object for signals.

Authors:
    Jean-Baptiste Bayle <j2b.bayle@gmail.com>
"""

import abc
import logging
import h5py
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ForEachObject(abc.ABC):
    """Abstract class which represents a dictionary holding a value for each object."""

    def __init__(self, values):
        """Initialize an object with a value or a function of the mosa index.

        Args:
            values: a value, a dictionary of values, or a function (mosa -> value)
        """
        if isinstance(values, dict):
            self.dict = {mosa: values[mosa] for mosa in self.indices()}
        elif callable(values):
            self.dict = {mosa: values(mosa) for mosa in self.indices()}
        elif isinstance(values, h5py.Dataset):
            self.dict = {mosa: values[mosa] for mosa in self.indices()}
        else:
            self.dict = {mosa: values for mosa in self.indices()}

    @classmethod
    @abc.abstractmethod
    def indices(cls):
        """Return list of object indices."""
        raise NotImplementedError

    def transformed(self, transformation):
        """Return a new dictionary from transformed objects.

        Args:
            transformation: function (mosa, value -> new_value)
        """
        return self.__class__(lambda mosa: transformation(mosa, self[mosa]))

    def collapsed(self):
        """Turn a numpy arrays containing identical elements into a scalar.

        This method can be used to optimize computations when constant time series are involved.
        """
        return self.transformed(lambda _, x:
            x[0] if isinstance(x, np.ndarray) and np.all(x == x[0]) else x
        )

    def write(self, hdf5, dataset):
        """Write values in dataset on HDF5 file.

        Args:
            hdf5: HDF5 file in which to write
            dataset: dataset name or path
        """
        # Retreive the maximum size of data
        size = 1
        for value in self.values():
            if np.isscalar(value):
                continue
            if size != 1 and len(value) != size:
                raise ValueError(f"incompatible sizes in dictionary '{size}' and '{len(size)}'")
            size = max(size, len(value))
        logger.debug("Writing dataset of size '%s' with '%s' columns", size, len(self.indices()))
        # Write dataset
        dtype = np.dtype({'names': self.indices(), 'formats': len(self.indices()) * [np.float64]})
        hdf5.create_dataset(dataset, (size,), dtype=dtype)
        for index in self.indices():
            hdf5[dataset][index] = self[index]

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, item):
        self.dict[key] = item

    def values(self):
        """Return dictionary values."""
        return self.dict.values()

    def keys(self):
        """Return dictionary keys."""
        return self.dict.keys()

    def items(self):
        """Return dictionary items."""
        return self.dict.items()

    def plot(self, output=None, dt=1, t0=0, size='auto', title='Signals'):
        """Plot signals for each object.

        Args:
            output: output file, None to show the plots
            dt: sampling period [s]
            t0: initial time [s]
            size: duration of time series, or 'auto' [samples]
            title: plot title
        """
        if size == 'auto':
            size = len(self) if len(self) > 1 else 100
        t = t0 + np.arange(size) * dt
        # Plot signals
        logger.info("Plotting signals for each object")
        plt.figure(figsize=(12, 4))
        for key, signal in self.items():
            plt.plot(t, np.broadcast_to(signal, size), label=key)
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Signal")
        plt.title(title)
        # Save or show glitch
        if output is not None:
            logger.info("Saving plot to %s", output)
            plt.savefig(output, bbox_inches='tight')
        else:
            plt.show()

    def __len__(self):
        """Return maximum size of signals."""
        sizes = [1 if np.isscalar(signal) else len(signal) for signal in self.values()]
        return np.max(sizes)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.dict == other.dict
        if isinstance(other, dict):
            return self.dict == other
        return np.all([self[index] == other for index in self.indices()])

    def __abs__(self):
        return self.transformed(lambda index, value: abs(value))

    def __neg__(self):
        return self.transformed(lambda index, value: -value)

    def __add__(self, other):
        if isinstance(other, ForEachObject):
            if isinstance(other, type(self)):
                return self.transformed(lambda index, value: value + other[index])
            raise TypeError(f"unsupported operand type for +: '{type(self)}' and '{type(other)}'")
        return self.transformed(lambda _, value: value + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ForEachObject):
            if isinstance(other, type(self)):
                return self.transformed(lambda index, value: value - other[index])
            raise TypeError(f"unsupported operand type for -: '{type(self)}' and '{type(other)}'")
        return self.transformed(lambda _, value: value - other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        if isinstance(other, ForEachObject):
            if isinstance(other, type(self)):
                return self.transformed(lambda index, value: value * other[index])
            raise TypeError(f"unsupported operand type for *: '{type(self)}' and '{type(other)}'")
        return self.transformed(lambda _, value: value * other)

    def __rmul__(self, other):
        return self * other

    def __floordiv__(self, other):
        if isinstance(other, ForEachObject):
            if isinstance(other, type(self)):
                return self.transformed(lambda index, value: value // other[index])
            raise TypeError(f"unsupported operand type for //: '{type(self)}' and '{type(other)}'")
        return self.transformed(lambda _, value: value // other)

    def __truediv__(self, other):
        if isinstance(other, ForEachObject):
            if isinstance(other, type(self)):
                return self.transformed(lambda index, value: value / other[index])
            raise TypeError(f"unsupported operand type for /: '{type(self)}' and '{type(other)}'")
        return self.transformed(lambda _, value: value / other)

    def __rtruediv__(self, other):
        return (self / other)**(-1)

    def __pow__(self, other):
        return self.transformed(lambda _, value: value**other)

    def __repr__(self):
        return repr(self.dict)


class ForEachSC(ForEachObject):
    """Represents a dictionary of values for each spacecraft."""

    @classmethod
    def indices(cls):
        return ['1', '2', '3']

    @staticmethod
    def distant_left_sc(sc):
        """Return index of distant rleftspacecraft."""
        if sc not in ForEachSC.indices():
            raise ValueError(f"invalid spacecraft index '{sc}'")
        return f'{int(sc) % 3 + 1}'

    @staticmethod
    def distant_right_sc(sc):
        """Return index of distant right spacecraft."""
        if sc not in ForEachSC.indices():
            raise ValueError(f"invalid spacecraft index '{sc}'")
        return f'{(int(sc) - 2) % 3 + 1}'

    @staticmethod
    def left_mosa(sc):
        """Return index of left MOSA."""
        if sc not in ForEachSC.indices():
            raise ValueError(f"invalid spacecraft index '{sc}'")
        return f'{sc}{ForEachSC.distant_left_sc(sc)}'

    @staticmethod
    def right_mosa(sc):
        """Return index of right MOSA."""
        if sc not in ForEachSC.indices():
            raise ValueError(f"invalid spacecraft index '{sc}'")
        return f'{sc}{ForEachSC.distant_right_sc(sc)}'

    def for_each_mosa(self):
        """Return a ForEachMOSA instance by sharing the spacecraft values on both MOSAs."""
        return ForEachMOSA(lambda mosa: self[ForEachMOSA.sc(mosa)])

    def __add__(self, other):
        if isinstance(other, ForEachMOSA):
            return self.for_each_mosa() + other
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, ForEachMOSA):
            return self.for_each_mosa() - other
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, ForEachMOSA):
            return self.for_each_mosa() * other
        return super().__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, ForEachMOSA):
            return self.for_each_mosa() // other
        return super().__floordiv__(other)

    def __truediv__(self, other):
        if isinstance(other, ForEachMOSA):
            return self.for_each_mosa() / other
        return super().__truediv__(other)


class ForEachMOSA(ForEachObject):
    """Represents a dictionary of values for each moveable optical subassembly (MOSA)."""

    @classmethod
    def indices(cls):
        return ['12', '23', '31', '13', '32', '21']

    @staticmethod
    def sc(mosa):
        """Return index of spacecraft hosting MOSA."""
        return f'{mosa[0]}'

    @staticmethod
    def distant_mosa(mosa):
        """Return index of distant MOSA.

        In practive, we invert the indices to swap emitter and receiver.
        """
        if mosa not in ForEachMOSA.indices():
            raise ValueError(f"invalid MOSA index '{mosa}'")
        return f'{mosa[1]}{mosa[0]}'

    @staticmethod
    def adjacent_mosa(mosa):
        """Return index of adjacent MOSA.

        In practice, we replace the second index by the only unused spacecraft index.
        """
        if mosa not in ForEachMOSA.indices():
            raise ValueError(f"invalid MOSA index '{mosa}'")
        unused = [sc for sc in ForEachSC.indices() if sc not in mosa]
        if len(unused) != 1:
            raise RuntimeError(f"cannot find adjacent MOSA for '{mosa}'")
        return f'{mosa[0]}{unused[0]}'

    def distant(self):
        """Return a ForEachMOSA instance for distant MOSAs."""
        return ForEachMOSA(lambda mosa: self[ForEachMOSA.distant_mosa(mosa)])

    def adjacent(self):
        """Return a ForEachMOSA instance for adjacent MOSAs."""
        return ForEachMOSA(lambda mosa: self[ForEachMOSA.adjacent_mosa(mosa)])

    def __add__(self, other):
        if isinstance(other, ForEachSC):
            return self + other.for_each_mosa()
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, ForEachSC):
            return self - other.for_each_mosa()
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, ForEachSC):
            return self * other.for_each_mosa()
        return super().__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, ForEachSC):
            return self // other.for_each_mosa()
        return super().__floordiv__(other)

    def __truediv__(self, other):
        if isinstance(other, ForEachSC):
            return self / other.for_each_mosa()
        return super().__truediv__(other)
