# By Bart Ouwehand 22-04-2022
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
from tqdm import tqdm

# Setup simluation parameters
fs = 0.05
day = 86400 # s
orig_duration = day*60 # X days

new_durations = np.array([1,5,10,30]) # X days

duration = day*new_durations
orig_size = orig_duration*fs
discard = 300
cutoff = 100
size = duration*fs-2*cutoff

create_fsdata = False

# Import the long data and export the new, short data
alpha = np.array([1,10,100,1000,10000])
N1 = 10
fp = 'measurements/tm_asds/'

print ("Exporting data")
for i,alph in enumerate(tqdm(alpha)):
    for j in range(N1):
        # If the original dataset only has sample data and not filtered sample data
        if create_fsdata:
            inputf = fp+str(orig_duration//day)+'d/'+str(int(alph))+"/s"+str(j)+'.txt'
            rawdata = ascii.read(inputf)
            old_sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])
            
            tmp = []
            coeffs = scipy.signal.firls(73, bands=[0,1e-2,1.5e-2,fs/2], desired=[1,1,0,0],fs=fs)
            for k in range(1,3):
                fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=old_sdata[k],padlen=(old_sdata.shape[1]//2)+1)
                tmp.append(fdata_tmp[cutoff:-cutoff])
            old_fsdata = np.array([old_sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])
            
            filepath = 'measurements/tm_asds/'+str(orig_duration//day)+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
            filecontent = Table(old_fsdata.T, names=['t','A','E'])
            ascii.write(filecontent, filepath, overwrite=True)
        
        inputf = fp+str(orig_duration//day)+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
        rawdata = ascii.read(inputf)
        old_fsdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])
        
        for k,nd in enumerate(new_durations):
            new_fsdata = old_fsdata[:,:int(size[k])]
            
            outputf = fp+str(int(nd))+'d/'+str(int(alph))+"/fs"+str(j)+'.txt'
            filecontent = Table(new_fsdata.T, names=['t','A','E'])
            ascii.write(filecontent, outputf, overwrite=True)

print ("Data truncated!")


    
