# By Bart Ouwehand 09-06-2022

from my_functions import *

d,dr = -1,dur_range[-1]

if run_new_simulation:
    # Generate the Galactic binaries
    GenerateGalbins(orbit_path,gw_path,fs,size[d],Amp_true, f_true, phi0_true_forinst, gw_beta_true, gw_lambda_true, orbits_t0 + 1/fs)
    
    # Generate the instrument data
    # sAfunc, sEfunc, sTfunc = GenerateInstrumentAET(orbit_path, gw_path, fs, size, sample_outputf, discard)
    sAfunc, sEfunc = GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], sample_outputf, discard,noise=False)

# Retreive A, E, T data
rawdata = ascii.read(sample_outputf+'.txt')
# sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E'],rawdata['T']])
sdata = np.array([rawdata['t'],rawdata['A'],rawdata['E']])

# Plot raw data retreived from AET datastream
if gen_plots:
    for i in range(2):
        plt.plot(sdata[0]/day,sdata[i+1],label=rec[i],alpha=.5)
    plt.title('AET datastreams for '+str(dr)+'d simulation with all noises')
    plt.legend(loc=4)
    plt.xlabel('Time (d)')
    plt.ylabel('Amplitude')
    plt.savefig("plots/Sample_rawAETdata.jpg")

def psd_func(data):
    return scipy.signal.welch(data,fs=fs,window='nuttall',nperseg=len(data),detrend=False)

# Create psd for data
tmp = []
for i in range(1,3):
    ftmp, psdtmp = psd_func(sdata[i])
    tmp.append(psdtmp)
psd = np.array([ftmp,tmp[0],tmp[1]])

# Create filtered data
cutoff = 100
tmp = []
#coeffs = scipy.signal.firls(73,bands=[0,1,1.2,2],desired=[1,1,0,0],fs=fs)
#coeffs = scipy.signal.firls(73, bands=[0,1e-2,3e-2,5e-2], desired=[1,1,0,0],fs=fs)
coeffs = scipy.signal.firls(73, bands=[0,1e-2,1.5e-2,fs/2], desired=[1,1,0,0],fs=fs)
for i in range(1,3):
    fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=sdata[i],padlen=len(psd[0]))
    tmp.append(fdata_tmp[cutoff:-cutoff])
fsdata = np.array([sdata[0][cutoff:-cutoff],tmp[0],tmp[1]])

def indiv_mdata(orbit_path,gw_path,Afunc,Efunc,s,discard,cutoff):
    model_instru = Instrument(
        size=s, # in samples
        dt=1/fs,
        aafilter=('kaiser', 240, 0.275*fs, 0.725*fs),
        orbits=orbit_path, # realistic orbits (make sure it's consistent with glitches and GWs!)
        gws=gw_path,
    )
    model_instru.disable_all_noises()
    model_instru.simulate()
    
    rawdata = Data.from_instrument(model_instru)
    A = sAfunc(rawdata.measurements)[discard:]
    E = sEfunc(rawdata.measurements)[discard:]
    t = model_instru.t[discard:]
    
    return np.array([t,A,E])[:,cutoff:-cutoff]


# Ready to generate the model data!
fp = ext+'measurements/tm_asds/'+str(dr)+'d/'
fp2 = 'plots/'+str(dr)+'d/mdata/'

# TEMPORARILY TRY THIS METHOD
# N2, Amp_range = 1, [[1]]

# Generate the datafiles for the different binaries
for a, f, p, beta, lamb,i in zip(Amp_true, f_true, phi0_true, gw_beta_true, gw_lambda_true,range(Ngalbins)):
    
    # Amp_range = a*np.linspace(0.1,2.5,N2) # Range of amplitudes
    
    mdata_1b = np.zeros((1+N2*2,int(size[d]-500)))
    print ("Calculating mdata for binary {}".format(i+1))
    for j,Amp in enumerate(tqdm(Amp_range[i])):
    # for j,Amp in enumerate([1]):
        GenerateGalbins(orbit_path, gw_path, fs, size[d], [Amp], [f], [0], [beta], [lamb], orbits_t0 + 1/fs)
        
        # Try something different
        # mdata_tmp = indiv_mdata(orbit_path, gw_path,sAfunc, sEfunc, size[d], discard,cutoff)
        outputf = 'measurements/test_mdatav2'
        GenerateInstrumentAET(orbit_path, gw_path, fs, size[d], outputf, discard, False, sAfunc, sEfunc, noise=False)
        rawdata = ascii.read(outputf+'.txt')
        os.remove(outputf+'.txt')
        mdata_tmptmp = np.array([rawdata['t'],rawdata['A'],rawdata['E']])#[:,cutoff:-cutoff]
        
        # Filter sample data
        tmp = []
        for k in range(1,3):
            fdata_tmp = scipy.signal.filtfilt(coeffs,1., x=mdata_tmptmp[k],padlen=len(psd[0]))
            tmp.append(fdata_tmp[cutoff:-cutoff])
        mdata_tmp = np.array([mdata_tmptmp[0][cutoff:-cutoff],tmp[0],tmp[1]])
        
        mdata_1b[0] = mdata_tmp[0]
        mdata_1b[1+2*j:3+2*j] = mdata_tmp[1:]
    
    names = ['t']
    for k in range(N2):
        names.extend(['A'+str(k),'E'+str(k)])
    print ("datashape =",mdata_1b.shape)
    
    filename = fp+"mdata/binary_"+str(i)+".txt"
    # filecontent = Table(sdata.T, names=['t','A','E','T'])
    filecontent = Table(mdata_1b.T, names=names)
    ascii.write(filecontent, filename, overwrite=True)


    fig, axs = plt.subplots(2,sharex=True,figsize=(10,8),gridspec_kw={'hspace':0})
    # plt.plot(x,y,c='black',alpha=.8)
    for k in range(2):
        for j,Amp in enumerate(Amp_range[i]):
        # for j,Amp in enumerate([1]):
            axs[k].plot(mdata_1b[0]/day,mdata_1b[1+2*j+k],label='{:.3f} A$_0$'.format(Amp/a))
        axs[k].set_ylabel("{} amplitude".format(rec[k]))
    axs[0].set_title("Model data for binary {} at different amplitudes for duration {} d".format(i,dr))
    axs[0].legend(loc=1)
    axs[k].set_xlim(.2,.22)
    axs[k].set_xlabel("Time [d]")
    fig.savefig(fp2+"binary_"+str(i)+".jpg")
    plt.close(fig)

